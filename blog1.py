from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor

tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    pad_token="<|endoftext|>"  # 显式设置pad token
)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device_map="auto"
)

# 确保pad_token有效，若不存在则使用eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
device='cuda'
model.to(device)

# 设置生成参数
def generate_with_cot(prompt, max_length=1200):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    inputs["attention_mask"] = inputs.input_ids.ne(tokenizer.pad_token_id).int()
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        temperature=0.6, 
        top_p=0.85,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 生成函数（并行采样）
def generate_with_cot_parallel(prompts, max_length=1000, num_samples=3):
    """
    并行生成多个候选解
    :param prompts: 提示词列表（长度需与num_samples一致）
    :param num_samples: 并行采样数量
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=num_samples,  # 并行采样
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

# 顺序修订函数
def joint_optimization_solver(problem, num_parallel=3, max_revision=2):
    visualizer = OptimizationVisualizer()
    
    # 并行生成初始候选解
    parallel_prompts = [cot_prompt]*num_parallel
    candidates = generate_with_cot_parallel(parallel_prompts, num_samples=num_parallel)
    
    # 记录初始状态
    for cid, candidate in enumerate(candidates):
        x, y = parse_solution(candidate)
        visualizer.record_step(
            iteration=0,
            candidate_id=cid,
            x=x,
            y=y,
            stage='initial'
        )
    
    # 多线程顺序修订
    with ThreadPoolExecutor() as executor:
        futures = []
        for cid, candidate in enumerate(candidates):
            futures.append(executor.submit(
                revised_solution, 
                candidate,
                problem,
                max_revision,
                cid,
                visualizer  # 传入可视化器
            ))
        refined_solutions = [f.result() for f in futures]
    
    # 创建动画
    ani = animation.FuncAnimation(
        visualizer.fig, 
        visualizer.animate,
        frames=visualizer.history['iteration'].max()+1,
        interval=1000,
        blit=False
    )
    
    # 保存或显示动画
    ani.save('optimization_process.gif', writer='pillow')
    plt.show()
    
    return select_best_solution(refined_solutions)


# 联合优化主流程
def revised_solution(initial_sol, problem, max_revision, cid, visualizer):
    """带可视化记录的修订过程"""
    current_sol = initial_sol
    for iter in range(1, max_revision+1):
        x, y = parse_solution(current_sol)
        visualizer.record_step(
            iteration=iter,
            candidate_id=cid,
            x=x,
            y=y,
            stage='revised'
        )
        
        if validate_solution(x, y):
            break
        
        # 构建修订提示
        revision_prompt = f"""
        之前的解决方案可能有误，请重新检查：
        
        {problem}
        
        先前尝试：
        {current_sol}
        
        发现的潜在问题：
        - 计算结果不满足原方程
        - 推理步骤存在逻辑错误
        
        请逐步修正：
        """
        current_sol = generate_with_cot(revision_prompt)
    
    # 记录最终状态
    x_final, y_final = parse_solution(current_sol)
    visualizer.record_step(
        iteration=iter,
        candidate_id=cid,
        x=x_final,
        y=y_final,
        stage='final'
    )
    return current_sol


# 验证函数
def validate_solution(x, y):
    """验证解是否满足原方程"""
    if x is None or y is None:
        return False
    eq1 = abs((x + y) - 8) < 1e-3
    eq2 = abs((2*x - y) - 1) < 1e-3
    return eq1 and eq2

# 最优解选择策略
def select_best_solution(solutions):
    """简单选择第一个有效解，可扩展为投票机制"""
    for sol in solutions:
        x, y = parse_solution(sol)
        if validate_solution(x, y):
            return sol
    return solutions[0]  # 降级策略


# 解析函数
def parse_solution(response):
    # 正则表达式匹配最终解
    solution_pattern = re.compile(
        r'(?:解为|答案|最终解)[^\d]*?'  # 匹配解的前导关键词
        r'x\s*[=＝]\s*([+-]?\d+\.?\d*)'  # 匹配x值，支持=和＝符号
        r'[^\d]*?'  # 非数字分隔符
        r'y\s*[=＝]\s*([+-]?\d+\.?\d*)',  # 匹配y值
        re.IGNORECASE | re.DOTALL
    )
    
    match = solution_pattern.search(response)
    if match:
        return float(match.group(1)), float(match.group(2))
    else:
        # 备用匹配：全局搜索最后一次出现的x=和y=
        x_matches = list(re.finditer(r'x\s*[=＝]\s*([+-]?\d+\.?\d*)', response))
        y_matches = list(re.finditer(r'y\s*[=＝]\s*([+-]?\d+\.?\d*)', response))
        if x_matches and y_matches:
            return float(x_matches[-1].group(1)), float(y_matches[-1].group(1))
        print("解析失败，请确保模型输出包含类似'x=3, y=5'的明确解")
        return None, None




# 新增可视化模块
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import pandas as pd

def plot_equations_solution(x_sol, y_sol):
    plt.figure(figsize=(8, 6))
    
    # 生成x值范围
    x = np.linspace(0, 10, 100)
    
    # 绘制方程1: x + y = 8 → y = 8 - x
    y1 = 8 - x
    plt.plot(x, y1, label='equation1: x + y = 8')
    
    # 绘制方程2: 2x - y = 1 → y = 2x - 1
    y2 = 2*x - 1
    plt.plot(x, y2, label='equation2: 2x - y = 1')
    
    # 标出解点
    plt.scatter(x_sol, y_sol, c='red', zorder=5, 
               label=f'Solution ({x_sol}, {y_sol})')
    
    plt.title("Visualization of systems of equations")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.axis([0, 10, 0, 10])
    plt.show()


class OptimizationVisualizer:
    def __init__(self):
        self.history = pd.DataFrame(columns=[
            'iteration', 'candidate_id', 'x', 'y', 
            'error_eq1', 'error_eq2', 'stage'
        ])
        self.fig = plt.figure(figsize=(15, 8))
        self.gs = GridSpec(3, 3, figure=self.fig)
        
        # 初始化子图
        self.ax1 = self.fig.add_subplot(self.gs[0:2, 0:2])  # 解空间分布
        self.ax2 = self.fig.add_subplot(self.gs[0:2, 2])    # 误差收敛曲线
        self.ax3 = self.fig.add_subplot(self.gs[2, :])      # 文本进度
        
        # 配置样式
        self.colors = plt.cm.viridis(np.linspace(0, 1, 5))
        self.stage_markers = {'initial': 'o', 'revised': 's', 'final': '*'}
    
    def record_step(self, iteration, candidate_id, x, y, stage):
        """记录优化过程中的每个步骤"""
        error_eq1 = abs(x + y - 8) if x and y else np.nan
        error_eq2 = abs(2*x - y - 1) if x and y else np.nan
        
        new_row = {
            'iteration': iteration,
            'candidate_id': candidate_id,
            'x': x,
            'y': y,
            'error_eq1': error_eq1,
            'error_eq2': error_eq2,
            'stage': stage
        }
        self.history = pd.concat([self.history, pd.DataFrame([new_row])], ignore_index=True)
    
    def plot_solution_space(self):
        """解空间动态分布图"""
        self.ax1.clear()
        
        # 绘制方程曲线
        x = np.linspace(0, 10, 100)
        self.ax1.plot(x, 8 - x, label='x + y = 8', alpha=0.5)
        self.ax1.plot(x, 2*x -1, label='2x - y = 1', alpha=0.5)
        
        # 绘制候选解轨迹
        for cid in self.history['candidate_id'].unique():
            df = self.history[self.history['candidate_id'] == cid]
            self.ax1.scatter(
                df['x'], df['y'], 
                c=df['iteration'], 
                cmap='viridis',
                marker=self.stage_markers[df['stage'].iloc[0]],
                label=f'Candidate {cid}'
            )
            self.ax1.plot(df['x'], df['y'], '--', alpha=0.3)
        
        self.ax1.set_title("Solution Space Evolution")
        self.ax1.legend()
        self.ax1.grid(True)
    
    def plot_error_convergence(self):
        """误差收敛曲线"""
        self.ax2.clear()
        
        # 计算总误差
        self.history['total_error'] = self.history['error_eq1'] + self.history['error_eq2']
        
        # 按迭代绘制误差
        for cid in self.history['candidate_id'].unique():
            df = self.history[self.history['candidate_id'] == cid]
            self.ax2.plot(
                df['iteration'], 
                df['total_error'],
                marker='o',
                label=f'Candidate {cid}'
            )
        
        self.ax2.set_yscale('log')
        self.ax2.set_title("Error Convergence")
        self.ax2.set_xlabel("Iteration")
        self.ax2.set_ylabel("Log Total Error")
        self.ax2.grid(True)
    
    def update_progress_text(self, current_iter):
        """文本进度显示"""
        self.ax3.clear()
        self.ax3.axis('off')
        
        latest = self.history[self.history['iteration'] == current_iter]
        text = []
        for _, row in latest.iterrows():
            text.append(
                f"Candidate {row['candidate_id']} ({row['stage']}):\n"
                f"x={row['x']:.2f}, y={row['y']:.2f}\n"
                f"Error: {row['total_error']:.2e}\n"
            )
        
        self.ax3.text(
            0, 0.5, 
            "\n".join(text), 
            fontfamily='monospace',
            verticalalignment='center'
        )
    
    def animate(self, i):
        """动画更新函数"""
        current_iter = self.history['iteration'].max()
        self.plot_solution_space()
        self.plot_error_convergence()
        self.update_progress_text(current_iter)
        return self.ax1, self.ax2, self.ax3




if __name__ == "__main__":

    problem = "解方程组：\n方程1: x + y = 8\n方程2: 2x - y = 1"

    # CoT提示：显式要求分步推理
    cot_prompt = f"""
    请逐步解决以下问题：

    {problem}

    分步推理：
    1. 观察方程组的结构，寻找消元方法。
    2. 通过相加方程消去变量y。
    3. 解出x的值。
    4. 代入求y。
    5. 验证解的正确性。

    """
    
    # 使用联合优化求解
    final_response = joint_optimization_solver(
        problem,
        num_parallel=3,
        max_revision=2
    )
    
    print("\n最终优化解：\n", final_response)
    
    # 可视化
    x_sol, y_sol = parse_solution(final_response)
    if x_sol is not None and y_sol is not None:
        plot_equations_solution(x_sol, y_sol)
    else:
        print("解析失败，请检查模型输出格式")