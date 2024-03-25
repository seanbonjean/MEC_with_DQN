from MEC_model import *
from DQN_mf import DeepQNetwork
import random
import copy
import xlwt
import matplotlib.pyplot as plt

TASK_NUM = 100


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
            user_list[user_index].queue_latency = 0.0  # 不考虑其他任务的影响，下同
            task.latency, task.energy = user_list[user_index].get_local_latency_energy(task.workload)
            self.local_values.append((task.latency, task.energy))

            # 全部本地MEC执行
            mec_list[server_index].queue_latency = 0.0
            task.latency, task.energy = mec_list[server_index].get_mec_latency_energy(task, speed, (
                user_index, server_index, server_index))
            self.local_mec_values.append((task.trans_latency, task.exe_latency, task.latency, task.energy))

            # 随机MEC执行
            # TODO 加入MEC协作后添加功能

            # 全部云上执行
            cloud.queue_latency = 0.0
            task.latency, task.energy = cloud.get_cloud_latency_energy(task, speed, (user_index, server_index))
            self.cloud_values.append((task.trans_latency, task.exe_latency, task.latency, task.energy))


class Optimal:
    """
    遍历所有情况寻找最优解，计算最优解下的total_cost
    """

    def __init__(self) -> None:
        self.total_cost = 0.0

    def get_opt_cost(self) -> None:
        for user in user_list:
            user.queue_latency = 0.0
        for mec in mec_list:
            mec.queue_latency = 0.0
        cloud.queue_latency = 0.0

        task_count = 0

        for task in task_list:
            user_index = task.my_user
            server_index = user_list[user_index].my_server

            # 本地执行
            task.execute_location = -1
            task.latency, task.energy = user_list[user_index].get_local_latency_energy(task.workload)
            task.get_cost()

            min_cost = task.cost  # cost的最小值
            min_cost_exe_loc = -1  # 最小cost对应的卸载位置
            min_cost_latency = task.latency  # 最小cost对应的时延
            min_cost_energy = task.energy  # 最小cost对应的能耗

            # 卸载到本地MEC
            task.execute_location = server_index  # TODO 加入MEC协作后更改为对所有MEC的遍历
            task.latency, task.energy = mec_list[task.execute_location].get_mec_latency_energy(task, speed, (
                user_index, server_index, task.execute_location))
            task.get_cost()

            if task.cost < min_cost:
                min_cost = task.cost
                min_cost_exe_loc = server_index
                min_cost_latency = task.latency
                min_cost_energy = task.energy

            # 卸载到云
            task.execute_location = -2
            task.latency, task.energy = cloud.get_cloud_latency_energy(task, speed, (user_index, server_index))
            task.get_cost()

            if task.cost < min_cost:
                min_cost = task.cost
                min_cost_exe_loc = -2
                min_cost_latency = task.latency
                min_cost_energy = task.energy

            task.execute_location = min_cost_exe_loc
            task.latency = min_cost_latency
            task.energy = min_cost_energy
            task.get_cost()

            if task.cost != min_cost:
                raise ValueError

            self.total_cost += task.cost

            # 任务队列的堆积
            if min_cost_exe_loc == -1:
                user_list[user_index].queue_latency = min_cost_latency
            elif min_cost_exe_loc == -2:
                cloud.queue_latency = min_cost_latency
            else:
                mec_list[min_cost_exe_loc].queue_latency = min_cost_latency

            # 经过一定时间，任务队列减小
            for user in user_list:
                user.queue_latency = max(user.queue_latency - task_interval[task_count], 0)
            for mec in mec_list:
                mec.queue_latency = max(mec.queue_latency - task_interval[task_count], 0)
            cloud.queue_latency = max(cloud.queue_latency - task_interval[task_count], 0)

            task_count += 1
        print(f"optimal total cost: {self.total_cost}")


class Greedy:
    """
    贪心算法的实现
    """

    def __init__(self) -> None:
        self.greedy_cost = list()  # 记录贪心算法下的任务cost，作为计算reward时的基准
        self.other_values = list()  # 记录卸载位置、时延、能耗等其他参数

    def get_greedy_results(self) -> None:
        for user in user_list:
            user.queue_latency = 0.0
        for mec in mec_list:
            mec.queue_latency = 0.0
        cloud.queue_latency = 0.0

        task_count = 0

        for task in task_list:
            user_index = task.my_user
            server_index = user_list[user_index].my_server

            option_list = list()  # 存储所有卸载选项

            # 本地执行
            option_list.append((-1, user_list[user_index].cpu_frequency, user_list[user_index].queue_latency))
            # 卸载到本地MEC
            i = server_index  # TODO 加入MEC协作后更改为遍历加入所有MEC
            option_list.append((i, mec_list[i].cpu_frequency, mec_list[i].queue_latency))
            # 卸载到云
            option_list.append((-2, cloud.cpu_frequency, cloud.queue_latency))

            first_option = list()  # 优先选队列为空的位置卸载
            last_option = list()
            for option in option_list:
                if option[2] < 0.1:
                    first_option.append(option)
                else:
                    last_option.append(option)
            if first_option:  # “首选”列表非空，即存在队列为空的位置
                # 选CPU频率最大的位置卸载
                first_option.sort(key=lambda x: x[1], reverse=True)
                task.execute_location = first_option[0][0]
            else:
                # 若无队列为空的位置，选队列时间最短的位置卸载
                last_option.sort(key=lambda x: x[2])
                task.execute_location = last_option[0][0]

            if task.execute_location == -1:  # 本地执行
                task.latency, task.energy = user_list[user_index].get_local_latency_energy(task.workload)
                user_list[user_index].queue_latency = task.latency
            elif task.execute_location == -2:  # 卸载到云
                task.latency, task.energy = cloud.get_cloud_latency_energy(task, speed, (user_index, server_index))
                cloud.queue_latency = task.latency
            else:  # 卸载到MEC
                task.latency, task.energy = mec_list[task.execute_location].get_mec_latency_energy(task, speed, (
                    user_index, server_index, task.execute_location))
                mec_list[task.execute_location].queue_latency = task.latency
            task.get_cost()

            self.greedy_cost.append(task.cost)
            self.other_values.append((task.execute_location, task.latency, task.energy))

            # 经过一定时间，任务队列减小
            for user in user_list:
                user.queue_latency = max(user.queue_latency - task_interval[task_count], 0)
            for mec in mec_list:
                mec.queue_latency = max(mec.queue_latency - task_interval[task_count], 0)
            cloud.queue_latency = max(cloud.queue_latency - task_interval[task_count], 0)

            task_count += 1
        print("****************************** greedy results ******************************")
        for i in range(len(self.greedy_cost)):
            print(f"execute location: {greedy.other_values[i][0]}, \t"
                  f"latency: {greedy.other_values[i][1]}, \t"
                  f"energy: {greedy.other_values[i][2]}, \t"
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
        user_list[user_index].queue_latency = task.latency
    elif task.execute_location == -2:  # 卸载到云
        task.latency, task.energy = cloud.get_cloud_latency_energy(task, speed, (user_index, server_index))
        cloud.queue_latency = task.latency
    else:  # 卸载到MEC
        task.latency, task.energy = mec_list[task.execute_location].get_mec_latency_energy(task, speed, (
            user_index, server_index, task.execute_location))
        mec_list[task.execute_location].queue_latency = task.latency
    task.get_cost()

    state[0] = task.cost
    state[1] -= 1

    task_count = TASK_NUM - remaining_task_num - 1
    baseline_cost = greedy.greedy_cost[task_count]
    reward = (baseline_cost - state[0]) / baseline_cost
    # reward = baseline_cost - state[0]

    done = remaining_task_num == 0

    # 经过一定时间，任务队列减小
    for user in user_list:
        user.queue_latency = max(user.queue_latency - task_interval[task_count], 0)
    for mec in mec_list:
        mec.queue_latency = max(mec.queue_latency - task_interval[task_count], 0)
    cloud.queue_latency = max(cloud.queue_latency - task_interval[task_count], 0)

    return state, reward, done, active_tasklist


class DQN:
    """
    维护强化学习的agent
    """

    def __init__(self) -> None:
        self.EPISODE_NUM = 200
        self.START_STEP = 300  # 第几步后开始学习
        self.INTERVAL_STEP = 1  # 每隔几步学习一次

        self.learning_rate = 0.0005  # 目前0.001效果比较好,0.0005比较好,0.0007,0.0008差别不大更好,0.0009好一些差别不大
        self.reward_decay = 0.7
        self.epsilon = 0.9
        self.replace_target_net_step = 100  # 每200步替换一次target_net参数，目前100比较好
        self.memory_size = 500  # 记忆库容量

        self.info_recorder = list()  # 记录每个episode中每一步的action/reward/cost（也就是state[0]）

        self.total_reward = list()  # 每个episode的累计reward
        self.total_cost = list()  # 每个episode所有任务的总成本

        n_actions = 3  # TODO 加入MEC协作后更改为 MEC_NUM + 2
        n_features = 2
        self.agent = DeepQNetwork(n_actions, n_features,
                                  learning_rate=self.learning_rate,
                                  reward_decay=self.reward_decay,
                                  e_greedy=self.epsilon,
                                  replace_target_iter=self.replace_target_net_step,
                                  memory_size=self.memory_size,
                                  output_graph=False  # 是否输出tensorboard文件
                                  )

    def train(self) -> None:
        step = 0  # 记录循环次数
        for episode in range(self.EPISODE_NUM):
            print(f"episode No. {episode}: ", end='')
            current_state, active_tasklist = state_init()
            total_reward = 0
            total_cost = 0
            done = False

            episode_info = list()

            while not done:
                step += 1
                # 选择action
                action = self.agent.choose_action(current_state)
                print(action, end=' ')

                # 执行action，环境变化
                next_state, reward, done, active_tasklist = state_step(current_state, action, active_tasklist)
                total_reward += reward
                total_cost += next_state[0]
                if done:
                    self.total_reward.append(total_reward)
                    self.total_cost.append(total_cost)

                # 提取各位置的任务队列长度
                qs_user = list()
                for i in range(USER_NUM):
                    qs_user.append(user_list[i].queue_latency)
                qs_mec = list()
                for i in range(MEC_NUM):
                    qs_mec.append(mec_list[i].queue_latency)
                qs_cloud = cloud.queue_latency
                queue_status = (qs_user, qs_mec, qs_cloud)
                # 保存该步所有信息，打印到excel
                episode_info.append((str(action), reward, next_state[0], queue_status))

                # 保存该条记忆
                self.agent.store_transition(current_state, action, reward, next_state)

                # 等待记忆库积累一定量内容后再开始学习
                if step > self.START_STEP and step % self.INTERVAL_STEP == 0:
                    self.agent.learn()

                current_state = next_state
            print()
            self.info_recorder.append(episode_info)


def make_excel(filename: str) -> None:
    workbook = xlwt.Workbook()

    worksheet = workbook.add_sheet("env")

    for i in range(4):
        worksheet.col(i).width = 256 * 16

    worksheet.write(0, 0, "user cpu freq")
    for i in range(USER_NUM):
        worksheet.write(i + 1, 0, user_list[i].cpu_frequency)

    worksheet.write(0, 1, "task.my_user")
    worksheet.write(0, 2, "task size")
    for i in range(TASK_NUM):
        worksheet.write(i + 1, 1, str(task_list[i].my_user))
        worksheet.write(i + 1, 2, task_list[i].data_size)

    worksheet.write(0, 3, "mec cpu freq")
    for i in range(MEC_NUM):
        worksheet.write(i + 1, 3, mec_list[i].cpu_frequency)

    worksheet = workbook.add_sheet("inspect")

    for i in range(17):
        worksheet.col(i).width = 256 * 12

    worksheet.write(0, 0, "local")
    worksheet.write(0, 2, "local mec")
    worksheet.write(0, 7, "random mec")
    worksheet.write(0, 12, "cloud")
    worksheet.write(1, 0, "latency")
    worksheet.write(1, 1, "energy")
    worksheet.write(1, 2, "trans u2m")
    worksheet.write(1, 3, "trans m2m")
    worksheet.write(1, 4, "exe latency")
    worksheet.write(1, 5, "latency")
    worksheet.write(1, 6, "energy")
    worksheet.write(1, 7, "trans u2m")
    worksheet.write(1, 8, "trans m2m")
    worksheet.write(1, 9, "exe latency")
    worksheet.write(1, 10, "latency")
    worksheet.write(1, 11, "energy")
    worksheet.write(1, 12, "trans u2m")
    worksheet.write(1, 13, "trans m2c")
    worksheet.write(1, 14, "exe latency")
    worksheet.write(1, 15, "latency")
    worksheet.write(1, 16, "energy")
    for i in range(TASK_NUM):
        worksheet.write(i + 2, 0, inspect.local_values[i][0])
        worksheet.write(i + 2, 1, inspect.local_values[i][1])
        worksheet.write(i + 2, 2, inspect.local_mec_values[i][0][0])
        worksheet.write(i + 2, 3, inspect.local_mec_values[i][0][1])
        worksheet.write(i + 2, 4, inspect.local_mec_values[i][1])
        worksheet.write(i + 2, 5, inspect.local_mec_values[i][2])
        worksheet.write(i + 2, 6, inspect.local_mec_values[i][3])
        # TODO 加入MEC协作后取消注释
        # worksheet.write(i + 2, 7, inspect.random_mec_values[i][0][0])
        # worksheet.write(i + 2, 8, inspect.random_mec_values[i][0][1])
        # worksheet.write(i + 2, 9, inspect.random_mec_values[i][1])
        # worksheet.write(i + 2, 10, inspect.random_mec_values[i][2])
        # worksheet.write(i + 2, 11, inspect.random_mec_values[i][3])
        worksheet.write(i + 2, 12, inspect.cloud_values[i][0][0])
        worksheet.write(i + 2, 13, inspect.cloud_values[i][0][1])
        worksheet.write(i + 2, 14, inspect.cloud_values[i][1])
        worksheet.write(i + 2, 15, inspect.cloud_values[i][2])
        worksheet.write(i + 2, 16, inspect.cloud_values[i][3])

    worksheet = workbook.add_sheet("greedy")

    for i in range(5):
        worksheet.col(i).width = 256 * 16
    for i in range(5, 7):
        worksheet.col(i).width = 256 * 20

    worksheet.write(0, 0, "execute location")
    worksheet.write(0, 1, "latency")
    worksheet.write(0, 2, "energy")
    worksheet.write(0, 3, "cost")
    for i in range(TASK_NUM):
        worksheet.write(i + 1, 0, str(greedy.other_values[i][0]))
        worksheet.write(i + 1, 1, greedy.other_values[i][1])
        worksheet.write(i + 1, 2, greedy.other_values[i][2])
        worksheet.write(i + 1, 3, greedy.greedy_cost[i])

    worksheet.write(1, 5, "optimal total cost")
    worksheet.write(1, 6, opt.total_cost)
    worksheet.write(2, 5, "greedy total cost")
    worksheet.write(2, 6, sum(greedy.greedy_cost))

    worksheet = workbook.add_sheet("DQN")

    for i in range(4):
        worksheet.col(i).width = 256 * 16
    for i in range(4, 6):
        worksheet.col(i).width = 256 * 20

    worksheet.write(0, 0, "episode No.")
    worksheet.write(0, 1, "total reward")
    worksheet.write(0, 2, "total cost")
    for i in range(dqn.EPISODE_NUM):
        worksheet.write(i + 1, 0, str(i + 1))
        worksheet.write(i + 1, 1, dqn.total_reward[i])
        worksheet.write(i + 1, 2, dqn.total_cost[i])

    worksheet.write(1, 4, "episode number")
    worksheet.write(2, 4, "start step")
    worksheet.write(3, 4, "interval step")
    worksheet.write(4, 4, "learning rate")
    worksheet.write(5, 4, "reward decay")
    worksheet.write(6, 4, "epsilon")
    worksheet.write(7, 4, "replace target net step")
    worksheet.write(8, 4, "memory size")
    worksheet.write(1, 5, dqn.EPISODE_NUM)
    worksheet.write(2, 5, dqn.START_STEP)
    worksheet.write(3, 5, dqn.INTERVAL_STEP)
    worksheet.write(4, 5, dqn.learning_rate)
    worksheet.write(5, 5, dqn.reward_decay)
    worksheet.write(6, 5, dqn.epsilon)
    worksheet.write(7, 5, dqn.replace_target_net_step)
    worksheet.write(8, 5, dqn.memory_size)

    for i in range(dqn.EPISODE_NUM):
        worksheet = workbook.add_sheet("epi No. " + str(i + 1))
        worksheet.write(0, 0, "total")
        worksheet.write(0, 2, dqn.total_reward[i])
        worksheet.write(0, 3, dqn.total_cost[i])
        worksheet.write(0, 5, "user No.")
        worksheet.write(0, 5 + USER_NUM, "mec No.")
        worksheet.write(0, 5 + USER_NUM + MEC_NUM, "cloud")
        worksheet.write(0, 6 + USER_NUM + MEC_NUM, "pass time")
        worksheet.write(1, 0, "step No.")
        worksheet.write(1, 1, "action")
        worksheet.write(1, 2, "reward")
        worksheet.write(1, 3, "cost")
        worksheet.write(1, 4, "queue")
        worksheet.write(2, 4, "status")
        for j in range(USER_NUM):
            worksheet.write(1, 5 + j, str(j + 1))
        for j in range(MEC_NUM):
            worksheet.write(1, 5 + USER_NUM + j, str(j + 1))

        for j in range(TASK_NUM):
            worksheet.write(2 + j, 0, str(j + 1))
            worksheet.write(2 + j, 1, dqn.info_recorder[i][j][0])
            worksheet.write(2 + j, 2, dqn.info_recorder[i][j][1])
            worksheet.write(2 + j, 3, dqn.info_recorder[i][j][2])
            for k in range(USER_NUM):
                worksheet.write(2 + j, 5 + k, dqn.info_recorder[i][j][3][0][k])
            for k in range(MEC_NUM):
                worksheet.write(2 + j, 5 + USER_NUM + k, dqn.info_recorder[i][j][3][1][k])
            worksheet.write(2 + j, 5 + USER_NUM + MEC_NUM, dqn.info_recorder[i][j][3][2])
            worksheet.write(2 + j, 6 + USER_NUM + MEC_NUM, task_interval[j])

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
    random.seed(1)

    user_list = [User(corresponding_MEC[i]) for i in range(USER_NUM)]
    task_list = [Task(random.randint(0, USER_NUM - 1)) for i in range(TASK_NUM)]
    # 任务之间的间隔时间
    MAX_INTERVAL = 0.5
    ZERO_PROBABILITY = 0.6
    task_interval = [random.uniform(0, MAX_INTERVAL) if random.uniform(0, 1) > ZERO_PROBABILITY else 0 for i in
                     range(TASK_NUM)]
    mec_list = [MecServer() for i in range(MEC_NUM)]
    cloud = CloudServer()
    speed = Channel()

    inspect = InspectParameters()
    inspect.try_all_situation()

    opt = Optimal()
    opt.get_opt_cost()

    greedy = Greedy()
    greedy.get_greedy_results()

    dqn = DQN()
    dqn.train()

    make_excel("C:/Users/sean-/Desktop/arguments.xls")
    plot_training_progress("C:/Users/sean-/Desktop/train.png")
