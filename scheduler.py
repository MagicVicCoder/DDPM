from diffusers import DDPMScheduler

def get_scheduler(num_timesteps=1000):
    return DDPMScheduler(
        num_train_timesteps=num_timesteps,
        beta_schedule='squaredcos_cap_v2'
    )
