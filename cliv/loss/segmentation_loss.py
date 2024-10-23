import torch

class CESegLoss(torch.nn.Module):

    def __init__(self,
                 ce_weights=None):
        
        super().__init__()

        if ce_weights is not None:
            self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(ce_weights))
        else:
            self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # have to squeeze a channel in target
        return self.loss(input, target.squeeze(1).long())
