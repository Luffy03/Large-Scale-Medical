import torch


def build_optimizer(args, model):
    ve_params = list(map(id, model.visual_extractor.parameters()))
    ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
    optimizer = getattr(torch.optim, args.optim)(
        [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
         {'params': ed_params, 'lr': args.lr_ed}],
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer


def build_lr_scheduler(args, optimizer):
    if args.lr_scheduler == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)
    elif args.lr_scheduler == "cosine":
        print(f"Using CosineAnnealingWarmRestarts lr_scheduler, T_0={args.step_size}, T_mult={args.gamma}")
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                            T_0=args.step_size,
                                                                            T_mult=int(args.gamma)
                                                                            )
    else:
        raise NotImplementedError

    return lr_scheduler
