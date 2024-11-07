from .Config_DF import ConfigDF


class Config:
    if ConfigDF.use_df_config:
        sequence_len = ConfigDF().sequence_len
    else:
        sequence_len = 16

    scale_factor = 0.64
    resolution = 128
    vae_latent_shape = [4, 32, 32]
    row_shape = [128 * 128 * 3, 1]
    data_start_end = [1, -(16 * 16 + 20)] # the useful line start and end point of training data
    data_shapes = [[sum(row_shape) * sequence_len]]


if __name__ == "__main__":
    print(Config.data_shapes[0])
    print(sum(Config.data_shapes[0]))
