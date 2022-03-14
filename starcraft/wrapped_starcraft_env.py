from smac.env.starcraft2.starcraft2 import StarCraft2Env
import numpy as np
from pysc2.lib import protocol
from absl import logging
from s2clientprotocol import sc2api_pb2 as sc_pb

class StarCraft2Env_MAF(StarCraft2Env):
    def __init__(self, *args, **kwargs):
        super(StarCraft2Env_MAF, self).__init__(*args, **kwargs)

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""
        actions_int = [int(a) for a in actions]

        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        # Collect individual actions
        sc_actions = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))

        for a_id, action in enumerate(actions_int):
            if not self.heuristic_ai:
                sc_action = self.get_agent_action(a_id, action)
            else:
                sc_action, action_num = self.get_agent_action_heuristic(
                    a_id, action)
                actions[a_id] = action_num
            if sc_action:
                sc_actions.append(sc_action)

        visibility_matrix_before_actions = self.get_visibility_matrix().copy()
        # Send action request
        req_actions = sc_pb.RequestAction(actions=sc_actions)
        try:
            self._controller.actions(req_actions)
            # Make step in SC2, i.e. apply actions
            self._controller.step(self._step_mul)
            # Observe here so that we know if the episode is over.
            self._obs = self._controller.observe()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            return 0, True, {}

        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        game_end_code = self.update_units()

        terminated = False
        reward = self.get_individual_rewards(visibility_matrix_before_actions, actions_int)
        info = {"battle_won": False}

        # count units that are still alive
        dead_allies, dead_enemies = 0, 0
        for al_id, al_unit in self.agents.items():
            if al_unit.health == 0:
                dead_allies += 1
        for e_id, e_unit in self.enemies.items():
            if e_unit.health == 0:
                dead_enemies += 1

        info['dead_allies'] = dead_allies
        info['dead_enemies'] = dead_enemies

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                info["battle_won"] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1

        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, '-'))

        if terminated:
            self._episode_count += 1

        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate

        return reward, terminated, info

    def get_individual_rewards(self, visibility, actions):
        """Reward function when self.reward_spare==False.
        Returns accumulative hit/shield point damage dealt to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to ally units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        """
        if self.reward_sparse:
            return np.zeros((self.n_agents))

        delta_deaths = 0
        delta_deaths_indv = np.zeros((self.n_agents))
        delta_ally = 0
        delta_ally_indv = np.zeros((self.n_agents))
        delta_enemy = 0
        delta_enemy_indv = np.zeros((self.n_agents))

        neg_scale = self.reward_negative_scale

        # update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:
                # did not die so far
                prev_health = (
                        self.previous_ally_units[al_id].health
                        + self.previous_ally_units[al_id].shield
                )
                if al_unit.health == 0:
                    # just died
                    self.death_tracker_ally[al_id] = 1
                    if not self.reward_only_positive:
                        delta_deaths_indv[al_id] -= self.reward_death_value * neg_scale
                        delta_deaths -= self.reward_death_value * neg_scale
                    delta_ally_indv[al_id] += prev_health * neg_scale
                    delta_ally += prev_health * neg_scale
                else:
                    # still alive
                    delta_ally_indv[al_id] += neg_scale * (
                            prev_health - al_unit.health - al_unit.shield
                    )
                    delta_ally += neg_scale * (
                            prev_health - al_unit.health - al_unit.shield
                    )

        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:
                prev_health = (
                        self.previous_enemy_units[e_id].health
                        + self.previous_enemy_units[e_id].shield
                )
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1

                    #we figure out credit assignment here:
                    agents_in_vicinity = np.arange(0, self.n_agents)[visibility[:, self.n_agents+e_id] == 1]
                    for agent in agents_in_vicinity:
                        if actions[agent] == (e_id + self.n_actions_no_attack):
                            delta_deaths_indv[agent] += self.reward_death_value
                            delta_enemy_indv[agent] += prev_health
                    delta_deaths += self.reward_death_value
                    delta_enemy += prev_health
                else:
                    agents_in_vicinity = np.arange(0, self.n_agents)[visibility[:, self.n_agents+e_id] == 1]
                    for agent in agents_in_vicinity:
                        if actions[agent] == (e_id + self.n_actions_no_attack):
                            delta_enemy_indv[agent] += prev_health - e_unit.health - e_unit.shield
                    delta_enemy += prev_health - e_unit.health - e_unit.shield

        if self.reward_only_positive:
            reward = np.abs(delta_enemy_indv + delta_deaths_indv)  # shield regeneration
        else:
            reward = delta_enemy_indv + delta_deaths_indv - delta_ally_indv

        return reward

env = StarCraft2Env()