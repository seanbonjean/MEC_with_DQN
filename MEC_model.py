import random
import numpy as np

USER_NUM = 25
MEC_NUM = 5

# 信道状态相关参数
u2m_band = 2.3 * (10 ** 6)  # 无线信道带宽 (Hz)
u2m_power = 1  # 无线传输功率 (W)
u2m_fade = 10 ** (-2)  # 功率损耗
u2m_noise = 10 ** (-6)  # 高斯噪声 (W)
m2m_power = 3  # 核心网传输功率m2m (W)
m2c_power = 3  # 核心网传输功率m2c (W)
m2c_speed = 1000.0  # 各基站到云的连接速率 (KB/s)

LATENCY_FACTOR = 0.18
ENERGY_FACTOR = 0.82

# USER_NUM = 25, MEC_NUM = 5
corresponding_MEC = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]

# USER_NUM = 25, MEC_NUM = 10
# corresponding_MEC = [4, 1, 7, 7, 4, 9, 8, 9, 5, 7, 4, 7, 9, 1, 3, 5, 4, 0, 8, 0, 2, 4, 6, 3, 0]


# USER_NUM = 25, MEC_NUM = 8
# corresponding_MEC = [5, 2, 7, 0, 6, 0, 2, 7, 4, 7, 7, 7, 7, 5, 0, 3, 5, 4, 5, 0, 1, 1, 3, 0, 5]

# corresponding_MEC = [0, 0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 5, 6, 6]  # USER_NUM = 17, MEC_NUM = 7
# corresponding_MEC = [0, 0, 1, 2, 3, 4, 4, 4, 5, 5, 6]  # MEC_NUM = 7
# corresponding_MEC = [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4]  # MEC_NUM = 5


class User:
    def __init__(self, server_index: int) -> None:
        self.cpu_frequency = random.uniform(1, 3)  # ED的计算能力，以CPU频率体现 (GHz)
        self.efficiency_factor = 0.03  # 芯片架构决定的能效因数，体现在能耗的计算中
        self.queue_latency = 0.0  # 清空当前任务队列所需的时间
        self.my_server = server_index  # 服务该用户的MEC服务器编号

    def get_local_latency_energy(self, workload: float) -> tuple:
        """
        任务本地执行，计算时延和能耗
        :param workload: 任务负载
        :returns latency, energy
        示例：
        task.latency, task.energy = user.get_local_latency_energy(task.workload)
        """
        exe_latency = workload / self.cpu_frequency
        latency = self.queue_latency + exe_latency

        energy = self.efficiency_factor * workload * self.cpu_frequency ** 2

        return latency, energy


class Task:
    def __init__(self, user_index: int) -> None:
        self.data_size = random.uniform(200, 600)  # 任务数据大小 (KB)
        self.cycles_per_bit = 800  # 每bit所需CPU循环次数
        self.workload = self.data_size * 2 ** 10 * 8 * self.cycles_per_bit * 10 ** (-9)  # 任务负载，以所需CPU循环次数体现 (G cycle)
        self.my_user = user_index  # 任务的生成者
        self.execute_location = 0  # 任务执行位置
        """
        本地执行：-1
        卸载到MEC：0~MEC_NUM-1
        卸载到云：-2
        接受action给出的卸载位置，以便核对是否在本地MEC执行
        因为action给出的是非负值，在接受action值时需要做一定转换，例如：
        if action == MEC_NUM:
            task.execute_location = -1
        elif action == MEC_NUM + 1:
            task.execute_location = -2
        else:
            task.execute_location = action
        """
        self.latency = 0.0  # 任务总时延
        self.trans_latency = (0.0, 0.0)  # 传输时延：第一跳u2m，第二跳m2m/m2c
        self.exe_latency = 0.0  # 执行时延
        self.energy = 0.0  # 任务能耗
        self.cost = 0.0  # 问题所优化的目标函数，是一个时延和能耗的加权参考值，将DQN每一episode的总cost与贪心算法得到的cost比较，以计算reward

    def get_cost(self) -> None:
        self.cost = self.latency * LATENCY_FACTOR + self.energy * ENERGY_FACTOR


class Channel:
    """
    信道状况，影响传输时延
    包括：用户到MEC、MEC之间、MEC到云
    """

    def __init__(self) -> None:
        self.u2m = np.empty([USER_NUM, MEC_NUM], dtype=float)  # user to MEC (KB/s)
        self.m2m = np.empty([MEC_NUM, MEC_NUM], dtype=float)  # MEC to MEC (KB/s)
        self.m2c = [m2c_speed for _ in range(MEC_NUM)]  # MEC to cloud (KB/s)

        # u2m初始化
        for i in range(USER_NUM):
            self.u2m[i, corresponding_MEC[i]] = u2m_band * np.log2(1 + u2m_power * u2m_fade / u2m_noise) / (8 * 2 ** 10)

        # m2m初始化
        for i in range(MEC_NUM):
            for j in range(MEC_NUM):
                # TODO 待调整数值
                self.m2m[i, j] = random.uniform(4000, 6000)


class MecServer:
    def __init__(self) -> None:
        self.cpu_frequency = random.randint(6, 8)  # MEC服务器的计算能力，以CPU频率体现 (GHz)
        self.queue_latency = 0.0  # 清空当前任务队列所需的时间

    def get_mec_latency_energy(self, task: Task, channel: Channel, indexes: tuple) -> tuple:
        """
        任务卸载到MEC，计算时延和能耗
        :param task: 任务
        :param channel: 信道状态
        :param indexes: 元组 (user, mec_from, mec_to)，表示任务在MEC系统中的数据传输方向；在本地MEC执行时，from、to皆为MEC自身序号
        :returns latency, energy
        """
        user = indexes[0]
        mec_from = indexes[1]
        mec_to = indexes[2]

        trans_latency_u2m = task.data_size / channel.u2m[user, mec_from]
        if mec_from != mec_to:
            trans_latency_m2m = task.data_size / channel.m2m[mec_from, mec_to]
        else:
            trans_latency_m2m = 0
        trans_latency = trans_latency_u2m + trans_latency_m2m
        exe_latency = task.workload / self.cpu_frequency
        latency = max(trans_latency, self.queue_latency) + exe_latency

        task.trans_latency = (trans_latency_u2m, trans_latency_m2m)
        task.exe_latency = exe_latency

        energy = u2m_power * trans_latency_u2m + m2m_power * trans_latency_m2m

        return latency, energy


class CloudServer:

    def __init__(self) -> None:
        self.cpu_frequency = 10  # 云的计算能力，以CPU频率体现 (GHz)
        self.queue_latency = 0.0  # 清空当前任务队列所需的时间

    def get_cloud_latency_energy(self, task: Task, channel: Channel, indexes: tuple) -> tuple:
        """
        任务卸载到云，计算时延和能耗
        :param task: 任务
        :param channel: 信道状态
        :param indexes: 元组 (user, mec)，表示任务在MEC系统中的数据传输方向
        :returns latency, energy
        """
        user = indexes[0]
        mec = indexes[1]

        trans_latency_u2m = task.data_size / channel.u2m[user, mec]
        trans_latency_m2c = task.data_size / channel.m2c[mec]
        trans_latency = trans_latency_u2m + trans_latency_m2c
        exe_latency = task.workload / self.cpu_frequency
        latency = max(trans_latency, self.queue_latency) + exe_latency

        task.trans_latency = (trans_latency_u2m, trans_latency_m2c)
        task.exe_latency = exe_latency

        energy = u2m_power * trans_latency_u2m + m2c_power * trans_latency_m2c

        return latency, energy
