import re

class ConfigDF():
    use_df_config = True

    def get_params(self, model_name):
        if self.use_dit:
            pattern = r'(?:_z(\d+))|(?:_c(\d+))|(?:_n(\d+))'
        else:
            pattern = r'(?:_z(\d+))|(?:_c(\d+))'
        matches = re.findall(pattern, model_name)
        numbers = []
        for match in matches:
            for num in match:
                if num:
                    numbers.append(int(num))
        return tuple(numbers)



    def __init__(self, model_name=None):

        self.dit_name_dict = {
            0: 'I64_S_2', 1: 'I64_S_4', 2: 'I64_S_8',
            3: 'I64_B_2', 4: 'I64_B_4', 5: 'I64_B_8',
            6: 'I64_L_2', 7: 'I64_L_4', 8: 'I64_L_8',
            9: 'I16_S_1', 10: 'I16_S_2', 11: 'I32_S_2',
            12: 'I32_B_2',
        }

        # for training setting
        if model_name is None:
            model_name = 'df_z32_c1_dit_n11_mario_km_tanh_ldm'

        print("model name:", model_name)
        self.model_name = model_name
        # model arch params
        self.use_dit = 'dit' in model_name
        self.use_mario = 'mario' in model_name
        self.use_km = 'km' in model_name
        self.use_tanh = 'tanh' in model_name
        self.use_lrelu = 'lrelu' in model_name
        self.use_mc = 'mc' in model_name


        params = self.get_params(model_name)
        self.zeta = params[0]
        self.context_len = params[1]
        if self.use_dit:
            self.dit_name = self.dit_name_dict[params[2]]



        # training params
        self.is_training = True
        self.lr = 1e-3
        self.wd = 0.002
        self.sequence_len = 48

        # fix params
        self.world_model_action_num = 46
        self.player_num = 1



