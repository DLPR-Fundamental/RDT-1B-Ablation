from collections import deque
class BasePolicyAdapter:
    required_history = 1
    policy_name = "base_policy"

    def reset(self):
        pass

    def infer(self, obs: dict):
        raise NotImplementedError

    def run_episode(self, env, max_steps=400, base_seed=0, episode_idx=0,
                    live_view=False, save_video=False):
        obs_window = deque(maxlen=self.required_history)
        bundle = env.reset(seed=episode_idx+base_seed)
        
        first_obs = self._extract_first_obs(bundle, env)
        for _ in range(self.required_history):
            obs_window.append(first_obs)

        self.reset()
        print(f"Bundle after reset: {bundle}")
        proprio = self._get_proprio(bundle)
        done = False
        steps = 0
        video_frames = []

        while steps < max_steps and not done:
            obs = self._prepare_obs(obs_window, proprio)
            actions_seq = self.infer(obs)
            for act in actions_seq:
                next_bundle, reward, terminated, truncated, info = env.step(act)
                proprio = self._get_proprio(next_bundle)
                next_obs = self._extract_obs(next_bundle, env)
                obs_window.append(next_obs)

                if live_view:
                    self._show_live(next_obs)
                if save_video:
                    video_frames.append(next_obs)

                steps += 1
                if terminated or truncated:
                    done = True
                    break

        success = info.get('success', False)
        return steps, success, video_frames

    def _extract_first_obs(self, bundle, env):
        raise NotImplementedError
    
    def _extract_obs(self, bundle, env):
        raise NotImplementedError

    def _get_proprio(self, bundle):
        raise NotImplementedError

    def _prepare_obs(self, obs_window, proprio):
        raise NotImplementedError

    def _show_live(self, img):
        import cv2
        cv2.imshow(f'{self.policy_name} Live', img[..., ::-1])
        cv2.waitKey(1)