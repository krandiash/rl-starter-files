import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_ac.utils.penv import ParallelEnv

import utils


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="how many worst episodes to show")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environments

envs = []
for i in range(args.procs):
    env = utils.make_env(args.env, args.seed + 10000 * i)
    envs.append(env)
env = ParallelEnv(envs)
print("Environments loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir, device, args.argmax, args.procs)
print("Agent loaded\n")

# Initialize logs

logs = {"num_frames_per_episode": [], "return_per_episode": [], "events_per_episode": []}

# Run agent

start_time = time.time()

obss = env.reset()

log_done_counter = 0
log_episode_return = torch.zeros(args.procs, device=device)
log_episode_num_frames = torch.zeros(args.procs, device=device)
log_events = [[] for _ in range(args.procs)]

while log_done_counter < args.episodes:
    actions = agent.get_actions(obss)
    obss, rewards, dones, infos = env.step(actions)
    agent.analyze_feedbacks(rewards, dones)

    for i in range(args.procs):
        log_events[i].append(infos[i]['events'])

    log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
    log_episode_num_frames += torch.ones(args.procs, device=device)

    for i, done in enumerate(dones):
        if done:
            log_done_counter += 1
            logs["return_per_episode"].append(log_episode_return[i].item())
            logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())
            logs["events_per_episode"].append(log_events[i])
            log_events[i] = []

    mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
    log_episode_return *= mask
    log_episode_num_frames *= mask

end_time = time.time()

# Print logs

num_frames = sum(logs["num_frames_per_episode"])
fps = num_frames/(end_time - start_time)
duration = int(end_time - start_time)
return_per_episode = utils.synthesize(logs["return_per_episode"])
num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
      .format(num_frames, fps, duration,
              *return_per_episode.values(),
              *num_frames_per_episode.values()))

# Print worst episodes

n = args.worst_episodes_to_show
if n > 0:
    print("\n{} worst episodes:".format(n))

    indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
    for i in indexes[:n]:
        print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))




# Print events
events = ['key_picked', 'left_door_1_passed', 'left_door_2_passed', 'locked_door_passed', 'key_dropped', 'ball_picked', 'distractor_key_picked', 'distractor_key_dropped']

goals_achieved = []
when_achieved = []
for ep_events in logs["events_per_episode"]:
    ep_events = np.array(ep_events)
    goals_achieved.append(np.max(ep_events, axis=0))

    when = np.argmax(ep_events, axis=0).astype(np.float)
    when[when == 0] = np.nan
    when_achieved.append(when)

print (goals_achieved)
print (when_achieved)

goals_achieved = np.array(goals_achieved)
when_achieved = np.array(when_achieved)
print ("\nevent success rates | steps to success:")
for i, event in enumerate(events):
    goal_stats = utils.synthesize(goals_achieved[:, i])
    when_stats = utils.synthesize(when_achieved[:, i])
    print("- {}:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | μσmM {:.2f} {:.2f} {:.2f} {:.2f}".format(events[i], *goal_stats.values(), *when_stats.values()))



