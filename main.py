from MEC_modle import *
# from DQN_mf import DeepQNetwork
import random

TASK_NUM = 100


class Standardize:
    """
    实现标准化处理的相关功能
    """

    def __init__(self) -> None:
        self.max = 0
        self.min = float('inf')

    def reset(self) -> None:
        self.max = 0
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
            # 全部本地执行
            latency, energy = user_list[task.my_user].get_local_latency_energy(task.workload)
            self.local_values.append((latency, energy))

            latency_standardizer.min_max_maintainer(latency)
            energy_standardizer.min_max_maintainer(energy)

            # 全部本地MEC执行
            user_index = task.my_user
            server_index = user_list[user_index].my_server
            latency, energy = mec_list[server_index].get_mec_latency_energy(task, speed,
                                                                            (user_index, server_index, server_index))
            self.local_mec_values.append((latency, energy))

            latency_standardizer.min_max_maintainer(latency)
            energy_standardizer.min_max_maintainer(energy)

            # 随机MEC执行
            # TODO 加入MEC协作后添加功能

            # 全部云上执行
            user_index = task.my_user
            server_index = user_list[user_index].my_server
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
            location_str = "随机MEC执行"
            values = self.random_mec_values
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
            cost = latency_std * latency_factor + energy_std * energy_factor
            cost_sum += cost
            if list_all:
                print(f"latency: {latency:<23}--std-->{latency_std:>10.3}, \t\t"
                      f"energy: {energy:<23}--std-->{energy_std:>10.3}, \t\t"
                      f"cost: {cost:<.3}")
        print(f"total cost: {cost_sum}")


if __name__ == "__main__":
    random.seed()

    user_list = [User(corresponding_MEC[i]) for i in range(USER_NUM)]
    task_list = [Task(random.randint(0, USER_NUM - 1)) for i in range(TASK_NUM)]
    mec_list = [MecServer() for i in range(MEC_NUM)]
    cloud = CloudServer()
    speed = Channel()

    latency_standardizer = Standardize()
    energy_standardizer = Standardize()

    inspect = InspectParameters()

    inspect.try_all_situation()
    inspect.show_situation("local", True)
    inspect.show_situation("local_mec", True)
    inspect.show_situation("random_mec", True)
    inspect.show_situation("cloud", True)
    latency_standardizer.reset()
    energy_standardizer.reset()
