import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from param import *
from utils import *
from spark_env.env import Environment
from average_reward import *
from compute_baselines import *
from compute_gradients import *
from actor_agent import ActorAgent
from tf_logger import TFLogger


def invoke_model(actor_agent, obs, exp):
    # parse observation
    job_dags, source_job, num_source_exec, \
    frontier_nodes, executor_limits, \
    exec_commit, moving_executors, action_map = obs

    if len(frontier_nodes) == 0:
        # no action to take
        return None, num_source_exec

    # invoking the learning model
    node_act, job_act, \
    node_act_probs, job_act_probs, \
    node_inputs, job_inputs, \
    node_valid_mask, job_valid_mask, \
    gcn_mats, gcn_masks, summ_mats, \
    running_dags_mat, dag_summ_backward_map, \
    exec_map, job_dags_changed = \
        actor_agent.invoke_model(obs)

    if sum(node_valid_mask[0, :]) == 0:
        # no node is valid to assign
        return None, num_source_exec

    # node_act should be valid
    assert node_valid_mask[0, node_act[0]] == 1

    # parse node action
    node = action_map[node_act[0]]

    # find job index based on node
    job_idx = job_dags.index(node.job_dag)

    # job_act should be valid
    assert job_valid_mask[0, job_act[0, job_idx] + \
                          len(actor_agent.executor_levels) * job_idx] == 1

    # find out the executor limit decision
    if node.job_dag is source_job:
        agent_exec_act = actor_agent.executor_levels[
                             job_act[0, job_idx]] - \
                         exec_map[node.job_dag] + \
                         num_source_exec
    else:
        agent_exec_act = actor_agent.executor_levels[
                             job_act[0, job_idx]] - exec_map[node.job_dag]

    # parse job limit action
    use_exec = min(
        node.num_tasks - node.next_task_idx - \
        exec_commit.node_commit[node] - \
        moving_executors.count(node),
        agent_exec_act, num_source_exec)

    # for storing the action vector in experience
    node_act_vec = np.zeros(node_act_probs.shape)
    node_act_vec[0, node_act[0]] = 1

    # for storing job index
    job_act_vec = np.zeros(job_act_probs.shape)
    job_act_vec[0, job_idx, job_act[0, job_idx]] = 1

    # store experience
    exp['node_inputs'].append(node_inputs)
    exp['job_inputs'].append(job_inputs)
    exp['summ_mats'].append(summ_mats)
    exp['running_dag_mat'].append(running_dags_mat)
    exp['node_act_vec'].append(node_act_vec)
    exp['job_act_vec'].append(job_act_vec)
    exp['node_valid_mask'].append(node_valid_mask)
    exp['job_valid_mask'].append(job_valid_mask)
    exp['job_state_change'].append(job_dags_changed)

    if job_dags_changed:
        exp['gcn_mats'].append(gcn_mats)
        exp['gcn_masks'].append(gcn_masks)
        exp['dag_summ_back_mat'].append(dag_summ_backward_map)

    return node, use_exec


# --exec_cap 8 --num_init_dags 1 --num_stream_dags 50 --reset_prob 5e-7 --reset_prob_min 5e-8 --reset_prob_decay 4e-10
# --diff_reward_enabled 1 --num_agents 2 --ba_size 8
# --model_save_interval 100 --model_folder models/stream_200_job_diff_reward_reset_5e-7_5e-8/
def train_agent(agent_id, param_queue, reward_queue, adv_queue, gradient_queue):
    # model evaluation seed
    # 当在代码中使用了随机数，但是希望代码在不同时间或者不同的机器上运行能够得到相同的随机数，以至于能够得到相同的结果，
    # 那么就需要到设置随机函数的seed数，对应的变量可以跨session生成相同的随机数。
    tf.set_random_seed(agent_id)

    # set up environment
    env = Environment()

    # gpu configuration
    config = tf.ConfigProto(device_count={'GPU': args.worker_num_gpu},
                            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.worker_gpu_fraction))

    sess = tf.Session(config=config)

    # set up actor agent
    actor_agent = ActorAgent(
        sess, args.node_input_dim, args.job_input_dim,
        args.hid_dims, args.output_dim, args.max_depth,
        range(1, args.exec_cap + 1))

    # collect experiences
    # 这是个死循环? 没有break语句，循环结束的条件是什么呢？
    while True:
        # get parameters from master; param_queue一个大小为1的队列，存储了当前agent的训练参数
        (actor_params, seed, max_time, entropy_weight) = param_queue.get()

        # synchronize model
        actor_agent.set_params(actor_params)

        # reset environment
        env.seed(seed)
        env.reset(max_time=max_time)

        # set up storage for experience
        exp = {'node_inputs': [], 'job_inputs': [], 'gcn_mats': [], 'gcn_masks': [],
               'summ_mats': [], 'running_dag_mat': [], 'dag_summ_back_mat': [],
               'node_act_vec': [], 'job_act_vec': [], 'node_valid_mask': [], 'job_valid_mask': [],
               'reward': [], 'wall_time': [], 'job_state_change': []}

        try:
            # The masking functions (node_valid_mask and job_valid_mask in actor_agent.py) has some
            # small chance (once in every few thousand iterations) to leave some non-zero probability
            # mass for a masked-out action.
            # This will trigger the check in "node_act and job_act should be valid" in actor_agent.py.
            # Whenever this is detected, we throw out the rollout of that iteration and try again.

            # run experiment
            obs = env.observe()
            done = False

            # initial time
            exp['wall_time'].append(env.wall_time.curr_time)
            '''
            done: 是指执行完一个step后的判断，若所有的job都已完成或者时间到了，则认为此agent的一个episode结束；
            为什么要等到一个episode结束？ 
            答： 因为此文是基于策略梯度训练的，在更新梯度的时候需要计算动作价值函数，
                 计算此动作价值函数的一个方法是，执行完一个epsisode，得到一个从s1,a1,r1, ... ,sT,aT,rT的轨迹，
                 通过该轨迹就可以计算出动作价值函数的值。详情可见DRL笔记，第三章，Policy-Based Reinforcement Learning
            '''
            while not done:
                node, use_exec = invoke_model(actor_agent, obs, exp)
                obs, reward, done = env.step(node, use_exec)

                if node is not None:
                    # valid action, store reward and time
                    exp['reward'].append(reward)
                    exp['wall_time'].append(env.wall_time.curr_time)
                elif len(exp['reward']) > 0:
                    # Note: if we skip the reward when node is None (i.e., no available actions),
                    # the sneaky agent will learn to exhaustively pick all nodes in one scheduling round,
                    # in order to avoid the negative reward.
                    exp['reward'][-1] += reward
                    exp['wall_time'][-1] = env.wall_time.curr_time

            # report reward signals to master
            assert len(exp['node_inputs']) == len(exp['reward'])
            '''
            在reward_queue中的就是main()中 result = reward_queue.get()
            result 为 batch_reward, batch_time, num_finished_jobs, avg_job_duration, reset_hit
            '''
            reward_queue.put(
                [exp['reward'], exp['wall_time'],
                 len(env.finished_job_dags),
                 np.mean([j.completion_time - j.start_time for j in env.finished_job_dags]),
                 env.wall_time.curr_time >= env.max_time])

            # get advantage term from master
            batch_adv = adv_queue.get()

            if batch_adv is None:
                # some other agents panic for the try and the main thread throw out the rollout,
                # reset and try again now.
                continue

            # compute gradients
            actor_gradient, loss = compute_actor_gradients(
                actor_agent, exp, batch_adv, entropy_weight)

            # report gradient to master
            gradient_queue.put([actor_gradient, loss])

        except AssertionError:
            # ask the main to abort this rollout and
            # try again
            reward_queue.put(None)
            # need to still get from adv_queue to 
            # prevent blocking
            adv_queue.get()


def main():
    # args是param.py文件中的变量，设置了很多参数的默认值。
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # create result and model folder
    create_folder_if_not_exists(args.result_folder)
    create_folder_if_not_exists(args.model_folder)

    # initialize communication queues
    # import multiprocessing as mp;
    # python在多进程方面提供了multiprocessing模块

    # params： 存储神经网络的参数
    # reward： 存储agent的奖励
    # gradient：存储训练时，模型更新的梯度
    # advantage：表达在状态s下，某动作a相对于平均而言的优势， 即batch_adv = all_cum_reward[i] - baselines[i]， 后文有
    params_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    reward_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    adv_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    gradient_queues = [mp.Queue(1) for _ in range(args.num_agents)]

    # set up training agents
    # num_agents， Number of parallel agents (given: 16)
    # 定义了一个存储args.num_agents个进程的列表
    # 每一个进程通过 mp.Process(target=, args=) 来定义
    agents = []
    for i in range(args.num_agents):
        # target表示被进程调用的函数名，arg表示传入该函数的参数
        agents.append(mp.Process(target=train_agent, args=(
            i, params_queues[i], reward_queues[i],
            adv_queues[i], gradient_queues[i])))

    # start training agents
    for i in range(args.num_agents):
        agents[i].start()

    # gpu configuration
    config = tf.ConfigProto(
        device_count={'GPU': args.master_num_gpu},
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=args.master_gpu_fraction))

    sess = tf.Session(config=config)
    # sess.run()

    # set up actor agent
    actor_agent = ActorAgent(
        # node_input_dim， node input dimensions to graph embedding (default: 5)
        # job_input_dim， job input dimensions to graph embedding (default: 3)
        # hid_dims， hidden dimensions throughout graph embedding (default: [16, 8])
        # output_dim，output dimensions throughout graph embedding (default: 8)
        # max_depth， Maximum depth of root-leaf message passing (default: 8)
        # exec_cap， Number of total executors (default: 100)
        sess, args.node_input_dim, args.job_input_dim,
        args.hid_dims, args.output_dim, args.max_depth,
        range(1, args.exec_cap + 1))

    # tensorboard logging
    tf_logger = TFLogger(sess, [
        'actor_loss', 'entropy', 'value_loss', 'episode_length',
        'average_reward_per_second', 'sum_reward', 'reset_probability',
        'num_jobs', 'reset_hit', 'average_job_duration',
        'entropy_weight'])

    # store average reward for computing differential rewards 存这个有啥用？
    # average_reward_storage_size， Storage size for computing average reward (default: 100000)
    # 只求固定数量(即，average_reward_storage_size)的reward的平均值
    avg_reward_calculator = AveragePerStepReward(args.average_reward_storage_size)

    # initialize entropy parameters
    # entropy_weight_init， Initial exploration entropy weight (default: 1)
    entropy_weight = args.entropy_weight_init

    # initialize episode reset probability
    # reset_prob， Probability for episode to reset (after x seconds) (given: 5e-7)
    # 5e-7 = 5乘10的负7次方
    reset_prob = args.reset_prob

    # ---- start training process ----
    # num_ep， Number of training epochs (default: 10000000)
    for ep in range(1, args.num_ep):
        print('training epoch', ep)

        # synchronize the model parameters for each training agent
        actor_params = actor_agent.get_params()

        # generate max time stochastically（随机地） based on reset prob = 5e-7
        # 会生成一个很大的正数，因为 reset prob = 5e-7 很小
        max_time = generate_coin_flips(reset_prob)

        # send out parameters to training agents
        for i in range(args.num_agents):
            # 问： params_queues[i]的大小为1，为什么每次put不会溢出？
            # 答： 不会，因为下面有 reward_queues[i].get()
            params_queues[i].put([actor_params, args.seed + ep,
                                  max_time, entropy_weight])

        # storage for advantage computation
        all_rewards, all_diff_times, all_times, \
        all_num_finished_jobs, all_avg_job_duration, \
        all_reset_hit, = [], [], [], [], [], []

        t1 = time.time()
        # get reward from agents
        any_agent_panic = False
        for i in range(args.num_agents):
            # get()， Remove and return an item from the queue
            # result 是指 batch_reward, batch_time, num_finished_jobs, avg_job_duration, reset_hit
            result = reward_queues[i].get()

            if result is None:
                # 第i个agent疯了，返回值不正常
                any_agent_panic = True
                continue
            else:
                '''
                ba_size: Batch size (default: 64)
                batch_reward: batch_reward是一个列表，存储了数量为batch的reward，此batch的大小是指ba_size吗? 一个batch包含64个job？
                batch_time: ?
                num_finished_jobs: 已完成的job的数量
                avg_job_duration： 平均每个job完成的时间
                reset_hit： 什么叫reset？这是一个什么样的动作？
                '''
                batch_reward, batch_time, \
                num_finished_jobs, avg_job_duration, \
                reset_hit = result

            '''
            diff_time: 是一个时间差，batch_time的第一个元素值减去最后一个元素值
            '''
            diff_time = np.array(batch_time[1:]) - np.array(batch_time[:-1])

            all_rewards.append(batch_reward)
            all_diff_times.append(diff_time)
            all_times.append(batch_time[1:])
            all_num_finished_jobs.append(num_finished_jobs)
            all_avg_job_duration.append(avg_job_duration)
            all_reset_hit.append(reset_hit)

            avg_reward_calculator.add_list_filter_zero(batch_reward, diff_time)

        t2 = time.time()
        print('got reward from workers', t2 - t1, 'seconds')

        if any_agent_panic:
            # The try condition breaks in some agent (should
            # happen rarely), throw out this rollout and try
            # again for next iteration (TODO: log this event)
            for i in range(args.num_agents):
                adv_queues[i].put(None)
            continue

        # compute differential reward
        all_cum_reward = []
        avg_per_step_reward = avg_reward_calculator.get_avg_per_step_reward()
        for i in range(args.num_agents):
            if args.diff_reward_enabled:
                # differential reward mode on
                rewards = np.array([r - avg_per_step_reward * t for (r, t) in zip(all_rewards[i], all_diff_times[i])])
            else:
                # regular reward
                rewards = np.array([r for (r, t) in zip(all_rewards[i], all_diff_times[i])])

            cum_reward = discount(rewards, args.gamma)

            all_cum_reward.append(cum_reward)

        # compute baseline
        baselines = get_piecewise_linear_fit_baseline(all_cum_reward, all_times)

        # give worker back the advantage
        for i in range(args.num_agents):
            batch_adv = all_cum_reward[i] - baselines[i]
            batch_adv = np.reshape(batch_adv, [len(batch_adv), 1])
            adv_queues[i].put(batch_adv)

        t3 = time.time()
        print('advantage ready', t3 - t2, 'seconds')

        actor_gradients = []
        all_action_loss = []  # for tensorboard
        all_entropy = []  # for tensorboard
        all_value_loss = []  # for tensorboard

        for i in range(args.num_agents):
            (actor_gradient, loss) = gradient_queues[i].get()

            actor_gradients.append(actor_gradient)
            all_action_loss.append(loss[0])
            all_entropy.append(-loss[1] / float(all_cum_reward[i].shape[0]))
            all_value_loss.append(loss[2])

        t4 = time.time()
        print('worker send back gradients', t4 - t3, 'seconds')

        actor_agent.apply_gradients(aggregate_gradients(actor_gradients), args.lr)

        t5 = time.time()
        print('apply gradient', t5 - t4, 'seconds')

        tf_logger.log(ep, [
            np.mean(all_action_loss),
            np.mean(all_entropy),
            np.mean(all_value_loss),
            np.mean([len(b) for b in baselines]),
            avg_per_step_reward * args.reward_scale,
            np.mean([cr[0] for cr in all_cum_reward]),
            reset_prob,
            np.mean(all_num_finished_jobs),
            np.mean(all_reset_hit),
            np.mean(all_avg_job_duration),
            entropy_weight])

        # decrease entropy weight
        entropy_weight = decrease_var(entropy_weight, args.entropy_weight_min, args.entropy_weight_decay)

        # decrease reset probability
        reset_prob = decrease_var(reset_prob, args.reset_prob_min, args.reset_prob_decay)

        if ep % args.model_save_interval == 0:
            actor_agent.save_model(args.model_folder + 'model_ep_' + str(ep))

    sess.close()


if __name__ == '__main__':
    main()
