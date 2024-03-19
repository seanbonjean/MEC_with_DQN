from MEC_modle import *
# from DQN_mf import DeepQNetwork
import random

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

        cost_sum = 0.0

        for latency, energy in values:
            latency_std = latency_standardizer.get_standardized_value(latency)
            energy_std = energy_standardizer.get_standardized_value(energy)
            cost = get_cost_external(latency_std, energy_std)
            cost_sum += cost
            if list_all:
                print(f"latency: {latency:<23}--std-->{latency_std:>10.3}, \t\t"
                      f"energy: {energy:<23}--std-->{energy_std:>10.3}, \t\t"
                      f"cost: {cost:<.3}")
        print(f"total cost: {cost_sum}")


class Greedy:
    """
    贪心算法的实现
    """

    def __init__(self) -> None:
        self.greedy_cost = list()  # 记录贪心算法下的任务cost，作为计算reward时的基准
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

    def show_greedy_results(self) -> None:
        print("******************************贪心卸载结果******************************")
        for i in range(len(self.greedy_cost)):
            print(f"execute location: {greedy.other_values[i][0]}, \t\t"
                  f"latency: {greedy.other_values[i][1]}, \t\t"
                  f"energy: {greedy.other_values[i][2]}, \t\t"
                  f"cost: {greedy.greedy_cost[i]}")


def state_init() -> None:
    pass


def state_step() -> None:
    # 千万注意老师的代码里greedy_list是cost的累加值，而我这里greedy_cost是对应值，算base的时候记得要累加！
    pass


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
    inspect.show_situation("local", False)
    inspect.show_situation("local_mec", False)
    inspect.show_situation("random_mec", False)
    inspect.show_situation("cloud", False)
    # latency_standardizer.reset()
    # energy_standardizer.reset()

    greedy = Greedy()
    greedy.get_greedy_results()
    greedy.show_greedy_results()
