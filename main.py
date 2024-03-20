from MEC_model import *
from DQN_mf import DeepQNetwork
import random
import copy
import xlwt
import matplotlib.pyplot as plt

TASK_NUM = 100


class Standardizer:
    """
    实现标准化处理的相关功能
    """

    def __init__(self) -> None:
        self.max = 0.0
        self.min = float('inf')

    def reset(self) -> None:
        self.max = 0.0
        self.min = float('inf')

    def min_max_maintainer(self, new_sample: float) -> None:
        """
        维护最大最小值
        """
        if self.max < new_sample:
            self.max = new_sample
        if self.min > new_sample:
            self.min = new_sample

    def get_standardized_value(self, original: float) -> float:
        """
        进行标准化处理
        """
        return (original - self.min) / (self.max - self.min)


class InspectParameters:
    """
    检视当前参数对MEC系统性能的影响
    """

    def __init__(self) -> None:
        """
        模拟执行时，MEC系统体现出的的时延和能耗性能，将会保存在这里
        """
        self.local_values = list()
        self.local_mec_values = list()
        self.random_mec_values = list()
        self.cloud_values = list()

    def try_all_situation(self) -> None:
        """
        执行所有情景，保存各自的时延和能耗性能，并维护标准化所需的最大最小值
        """
        for task in task_list:
            user_index = task.my_user
            server_index = user_list[user_index].my_server

            # 全部本地执行
            latency, energy = user_list[user_index].get_local_latency_energy(task.workload)
            self.local_values.append((latency, energy))

            latency_standardizer.min_max_maintainer(latency)
            energy_standardizer.min_max_maintainer(energy)

            # 全部本地MEC执行
            latency, energy = mec_list[server_index].get_mec_latency_energy(task, speed,
                                                                            (user_index, server_index, server_index))
            self.local_mec_values.append((latency, energy))

            latency_standardizer.min_max_maintainer(latency)
            energy_standardizer.min_max_maintainer(energy)

            # 随机MEC执行
            # TODO 加入MEC协作后添加功能

            # 全部云上执行
            latency, energy = cloud.get_cloud_latency_energy(task, speed, (user_index, server_index))
            self.cloud_values.append((latency, energy))

            latency_standardizer.min_max_maintainer(latency)
            energy_standardizer.min_max_maintainer(energy)

    def show_situation(self, location: str, list_all: bool) -> None:
        """
        进行标准化处理，计算成本和总成本，显示所选情景的所有性能参数
        :param location: 选择情景
        :param list_all: 选择是否列出所有task对应的性能参数
        """
        if location == "local":
            location_str = "全部本地执行"
            values = self.local_values
        elif location == "local_mec":
            location_str = "全部本地MEC执行"
            values = self.local_mec_values
        elif location == "random_mec":
            # TODO 加入MEC协作后添加功能
            # location_str = "随机MEC执行"
            # values = self.random_mec_values
            return
        elif location == "cloud":
            location_str = "全部云上执行"
            values = self.cloud_values
        else:
            raise ValueError

        if list_all:
            print("******************************" + location_str + "******************************")
        else:
            print(location_str, end=' ')

        total_cost = 0.0

        for latency, energy in values:
            latency_std = latency_standardizer.get_standardized_value(latency)
            energy_std = energy_standardizer.get_standardized_value(energy)
            cost = get_cost_external(latency_std, energy_std)
            total_cost += cost
            if list_all:
                print(f"latency: {latency:<23}--std-->{latency_std:>10.3}, \t\t"
                      f"energy: {energy:<23}--std-->{energy_std:>10.3}, \t\t"
                      f"cost: {cost:<.3}")
        print(f"total cost: {total_cost}")


class Greedy:
    """
    贪心算法的实现
    """

    def __init__(self) -> None:
        self.greedy_cost = list()  # 记录贪心算法下的任务cost，作为计算reward时的基准
        self.accumulated_cost = list()  # 累加了的cost，以便计算reward时调用
        self.other_values = list()  # 记录卸载位置、时延、能耗等其他参数

    def get_greedy_results(self) -> None:
        for task in task_list:
            user_index = task.my_user
            server_index = user_list[user_index].my_server

            # 本地执行
            task.execute_location = -1
            task.latency, task.energy = user_list[user_index].get_local_latency_energy(task.workload)
            task.latency_std = latency_standardizer.get_standardized_value(task.latency)
            task.energy_std = energy_standardizer.get_standardized_value(task.energy)
            task.get_cost()

            min_cost = task.cost  # cost的最小值
            min_cost_exe_loc = -1  # 最小cost对应的卸载位置
            min_cost_latency = task.latency  # 最小cost对应的时延
            min_cost_energy = task.energy  # 最小cost对应的能耗

            # 卸载到本地MEC
            task.execute_location = server_index  # TODO 加入MEC协作后更改为对所有MEC的遍历
            task.latency, task.energy = mec_list[task.execute_location].get_mec_latency_energy(task, speed, (
                user_index, server_index, task.execute_location))
            task.latency_std = latency_standardizer.get_standardized_value(task.latency)
            task.energy_std = energy_standardizer.get_standardized_value(task.energy)
            task.get_cost()

            if task.cost < min_cost:
                min_cost = task.cost
                min_cost_exe_loc = server_index
                min_cost_latency = task.latency
                min_cost_energy = task.energy

            # 卸载到云
            task.execute_location = -2
            task.latency, task.energy = cloud.get_cloud_latency_energy(task, speed, (user_index, server_index))
            task.latency_std = latency_standardizer.get_standardized_value(task.latency)
            task.energy_std = energy_standardizer.get_standardized_value(task.energy)
            task.get_cost()

            if task.cost < min_cost:
                min_cost = task.cost
                min_cost_exe_loc = -2
                min_cost_latency = task.latency
                min_cost_energy = task.energy

            task.execute_location = min_cost_exe_loc
            task.latency = min_cost_latency
            task.energy = min_cost_energy
            task.latency_std = latency_standardizer.get_standardized_value(task.latency)
            task.energy_std = energy_standardizer.get_standardized_value(task.energy)
            task.get_cost()

            if task.cost != min_cost:
                raise ValueError

            self.greedy_cost.append(min_cost)
            self.other_values.append((min_cost_exe_loc, min_cost_latency, min_cost_energy))

        accumulated_cost = 0
        for cost in self.greedy_cost:
            accumulated_cost += cost
            self.accumulated_cost.append(accumulated_cost)

    def show_greedy_results(self) -> None:
        print("******************************贪心卸载结果******************************")
        for i in range(len(self.greedy_cost)):
            print(f"execute location: {greedy.other_values[i][0]}, \t\t"
                  f"latency: {greedy.other_values[i][1]}, \t\t"
                  f"energy: {greedy.other_values[i][2]}, \t\t"
                  f"cost: {greedy.greedy_cost[i]}")


def state_init() -> tuple:
    """
    初始化系统状态，返回初始state和一份可供删改的任务列表副本
    """
    for user in user_list:
        user.queue_latency = 0.0

    for mec in mec_list:
        mec.queue_latency = 0.0

    cloud.queue_latency = 0.0

    for task in task_list:
        task.execute_location = 0
        task.latency = 0.0
        task.energy = 0.0
        task.latency_std = 0.0
        task.energy_std = 0.0
        task.cost = 0.0

    total_cost = 0
    initial_state = [total_cost, TASK_NUM]  # state的定义：已规划任务的总成本；未规划任务的数量
    active_tasklist = copy.deepcopy(task_list)  # 会被进行删改操作的、每一episode各自的tasklist

    return initial_state, active_tasklist


def state_step(state: list, action: int, active_tasklist: list) -> tuple:
    """
    提供系统状态变换相关功能，维护当前episode的tasklist副本 (active_tasklist)
    返回的done指示当前episode是否结束（所有任务是否全部卸载完毕）
    """
    task = active_tasklist.pop(0)
    remaining_task_num = len(active_tasklist)  # 剩余任务数（当前任务不计入）

    user_index = task.my_user
    server_index = user_list[user_index].my_server

    # TODO 加入MEC协作后更改为注释中内容
    if action == 0:
        task.execute_location = server_index
    elif action == 1:
        task.execute_location = -1
    elif action == 2:
        task.execute_location = -2
    else:
        raise ValueError
    """
    if action == MEC_NUM:
        task.execute_location = -1
    elif action == MEC_NUM + 1:
        task.execute_location = -2
    else:
        task.execute_location = action
    """

    if task.execute_location == -1:  # 本地执行
        task.latency, task.energy = user_list[user_index].get_local_latency_energy(task.workload)
    elif task.execute_location == -2:  # 卸载到云
        task.latency, task.energy = cloud.get_cloud_latency_energy(task, speed, (user_index, server_index))
    else:  # 卸载到MEC
        task.latency, task.energy = mec_list[task.execute_location].get_mec_latency_energy(task, speed, (
            user_index, server_index, task.execute_location))

    task.latency_std = latency_standardizer.get_standardized_value(task.latency)
    task.energy_std = energy_standardizer.get_standardized_value(task.energy)
    task.get_cost()

    state[0] += task.cost
    state[1] -= 1

    baseline_cost = greedy.accumulated_cost[TASK_NUM - remaining_task_num - 1]
    reward = (baseline_cost - state[0]) / baseline_cost

    done = remaining_task_num == 0

    return state, reward, done, active_tasklist


class DQN:
    """
    维护强化学习的agent
    """

    def __init__(self) -> None:
        self.EPISODE_NUM = 2000
        self.START_STEP = 20  # 第几步后开始学习
        self.INTERVAL_STEP = 1  # 每隔几步学习一次

        self.total_reward = list()  # 每个episode的累计reward
        self.total_cost = list()  # 每个episode所有任务的总成本

        n_actions = 3  # TODO 加入MEC协作后更改为 MEC_NUM + 2
        n_features = 2
        self.agent = DeepQNetwork(n_actions, n_features,
                                  learning_rate=0.0005,  # 目前0.001效果比较好,0.0005比较好,0.0007,0.0008差别不大更好,0.0009好一些差别不大
                                  reward_decay=0.7,  # 之前是0.7
                                  e_greedy=0.9,
                                  replace_target_iter=100,  # 每200步替换一次target_net参数，目前100比较好
                                  memory_size=500,  # 记忆上限
                                  output_graph=False  # 是否输出tensorboard文件
                                  )

    def train(self) -> None:
        step = 0  # 记录循环次数
        for episode in range(self.EPISODE_NUM):
            print(f"episode No. {episode}: ", end='')
            current_state, active_tasklist = state_init()
            total_reward = 0
            done = False

            while not done:
                step += 1
                # 选择action
                action = self.agent.choose_action(current_state)
                print(action, end=' ')

                # 执行action，环境变化
                next_state, reward, done, active_tasklist = state_step(current_state, action, active_tasklist)
                total_reward += reward
                if done:
                    self.total_reward.append(total_reward)
                    self.total_cost.append(next_state[0])

                # 保存该条记忆
                self.agent.store_transition(current_state, action, reward, next_state)

                # 等待记忆库积累一定量内容后再开始学习
                if step > self.START_STEP and step % self.INTERVAL_STEP == 0:
                    self.agent.learn()

                current_state = next_state
            print()


def make_excel(filename: str) -> None:
    workbook = xlwt.Workbook()

    worksheet = workbook.add_sheet("env")

    for i in range(4):
        worksheet.col(i).width = 256 * 20

    worksheet.write(0, 0, "user_cpu")
    for i in range(USER_NUM):
        worksheet.write(i + 1, 0, str(user_list[i].cpu_frequency))

    worksheet.write(0, 1, "task_size")
    worksheet.write(0, 2, "task_my_user")
    for i in range(TASK_NUM):
        worksheet.write(i + 1, 1, str(task_list[i].data_size))
        worksheet.write(i + 1, 2, str(task_list[i].my_user))

    worksheet.write(0, 3, "mec_cpu")
    for i in range(MEC_NUM):
        worksheet.write(i + 1, 3, str(mec_list[i].cpu_frequency))

    worksheet = workbook.add_sheet("inspect")

    for i in range(12):
        worksheet.col(i).width = 256 * 20

    worksheet.write(0, 0, "local")
    worksheet.write(0, 2, "local_mec")
    worksheet.write(0, 4, "random_mec")
    worksheet.write(0, 6, "cloud")
    for i in range(4):
        worksheet.write(1, i * 2, "latency")
        worksheet.write(1, i * 2 + 1, "energy")
    for i in range(TASK_NUM):
        worksheet.write(i + 2, 0, str(inspect.local_values[i][0]))
        worksheet.write(i + 2, 1, str(inspect.local_values[i][1]))
        worksheet.write(i + 2, 2, str(inspect.local_mec_values[i][0]))
        worksheet.write(i + 2, 3, str(inspect.local_mec_values[i][1]))
        # TODO 加入MEC协作后取消注释
        # worksheet.write(i + 2, 4, str(inspect.random_mec_values[i][0]))
        # worksheet.write(i + 2, 5, str(inspect.random_mec_values[i][1]))
        worksheet.write(i + 2, 6, str(inspect.cloud_values[i][0]))
        worksheet.write(i + 2, 7, str(inspect.cloud_values[i][1]))

    worksheet.write(0, 8, "standardizer")
    worksheet.write(1, 8, "latency")
    worksheet.write(1, 10, "energy")
    worksheet.write(2, 8, "min")
    worksheet.write(2, 9, "max")
    worksheet.write(2, 10, "min")
    worksheet.write(2, 11, "max")
    worksheet.write(3, 8, str(energy_standardizer.min))
    worksheet.write(3, 9, str(energy_standardizer.max))
    worksheet.write(3, 10, str(latency_standardizer.min))
    worksheet.write(3, 11, str(latency_standardizer.max))

    worksheet = workbook.add_sheet("greedy")

    for i in range(4):
        worksheet.col(i).width = 256 * 20

    worksheet.write(0, 0, "exe_loc")
    worksheet.write(0, 1, "latency")
    worksheet.write(0, 2, "energy")
    worksheet.write(0, 3, "cost")
    for i in range(TASK_NUM):
        worksheet.write(i + 1, 0, str(greedy.other_values[i][0]))
        worksheet.write(i + 1, 1, str(greedy.other_values[i][1]))
        worksheet.write(i + 1, 2, str(greedy.other_values[i][2]))
        worksheet.write(i + 1, 3, str(greedy.greedy_cost[i]))

    worksheet = workbook.add_sheet("DQN")

    for i in range(2):
        worksheet.col(i).width = 256 * 20

    worksheet.write(0, 0, "reward")
    worksheet.write(0, 1, "cost")
    for i in range(dqn.EPISODE_NUM):
        worksheet.write(i + 1, 0, str(dqn.total_reward[i]))
        worksheet.write(i + 1, 1, str(dqn.total_cost[i]))

    workbook.save(filename)


def plot_training_progress(filename: str) -> None:
    plt.subplot(2, 1, 1)
    plt.plot(dqn.total_reward)
    plt.ylabel("reward")

    plt.subplot(2, 1, 2)
    plt.plot(dqn.total_cost)
    plt.xlabel("episode")
    plt.ylabel("cost")

    plt.suptitle("training progress")
    plt.savefig(filename, dpi=300)
    plt.show()


if __name__ == "__main__":
    random.seed()

    user_list = [User(corresponding_MEC[i]) for i in range(USER_NUM)]
    task_list = [Task(random.randint(0, USER_NUM - 1)) for i in range(TASK_NUM)]
    mec_list = [MecServer() for i in range(MEC_NUM)]
    cloud = CloudServer()
    speed = Channel()

    latency_standardizer = Standardizer()
    energy_standardizer = Standardizer()

    inspect = InspectParameters()
    inspect.try_all_situation()
    inspect.show_situation("local", True)
    inspect.show_situation("local_mec", True)
    inspect.show_situation("random_mec", True)
    inspect.show_situation("cloud", True)
    # latency_standardizer.reset()
    # energy_standardizer.reset()

    greedy = Greedy()
    greedy.get_greedy_results()
    greedy.show_greedy_results()

    dqn = DQN()
    dqn.train()

    make_excel("C:/Users/sean-/Desktop/arguments.xls")
    plot_training_progress("C:/Users/sean-/Desktop/train.png")
