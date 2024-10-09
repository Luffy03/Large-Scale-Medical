from trainers.ssm_rot_trainer import RotTrainer

class RPLTrainer(RotTrainer):
    def __init__(self, config):
        super(RPLTrainer, self).__init__(config)
        assert config.model == 'Simple'

    def set_input(self, sample):
        uniform_patch, random_patch, target = sample
        self.uniform_patch = uniform_patch.to(self.device)
        self.input = random_patch.to(self.device)
        self.target = target.to(self.device)

    def forward(self):
        self.pred = self.model(self.uniform_patch, self.input)
