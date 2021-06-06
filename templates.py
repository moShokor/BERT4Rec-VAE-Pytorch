def set_template(args):
    if args.template is None:
        return

    elif args.template.startswith('train_bert'):
        args.mode = 'train'
        # TODO modify these args when adding a new dataset
        args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        #
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'leave_one_out'

        args.dataloader_code = 'bert'
        batch = 128
        # TODO maybe batch = 256
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch
        # TODO make sure the sampling is done based on popularity
        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100
        args.test_negative_sampling_seed = 98765

        args.trainer_code = 'bert'
        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        # TODO maybe args.lr = 0.0001
        args.lr = 0.001
        # TODO make sure the decay is linear, l2 weight decay of 0.01
        #  and that gradient clipping is applied if the l2 norm of the
        #  gradient is greater than 5, and the layers number is 2 and
        #  the number of heads is also 2
        args.enable_lr_schedule = True
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 100 if args.dataset_code == 'ml-1m' else 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]  # this is just the intervals to calculate the NDCG on
        args.best_metric = 'NDCG@10'

        args.model_code = 'bert'
        args.model_init_seed = 0

        args.bert_dropout = 0.1
        args.bert_hidden_units = 256
        # TODO this should be 0.2 args.bert_mask_prob = 0.2
        args.bert_mask_prob = 0.15
        args.bert_max_len = 100
        # TODO this should be 200 args.bert_max_len = 200
        args.bert_num_blocks = 2
        # TODO this should be 2 args.bert_num_heads = 4
        args.bert_num_heads = 4

    elif args.template.startswith('train_dae'):
        args.mode = 'train'

        args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'holdout'
        args.dataset_split_seed = 98765
        args.eval_set_size = 500 if args.dataset_code == 'ml-1m' else 10000

        args.dataloader_code = 'ae'
        batch = 128 if args.dataset_code == 'ml-1m' else 512
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.trainer_code = 'dae'
        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 1e-3
        args.enable_lr_schedule = False
        args.weight_decay = 0.00
        args.num_epochs = 100 if args.dataset_code == 'ml-1m' else 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.best_metric = 'NDCG@10'

        args.model_code = 'dae'
        args.model_init_seed = 0
        args.dae_num_hidden = 2
        args.dae_hidden_dim = 600
        args.dae_latent_dim = 200
        args.dae_dropout = 0.5

    elif args.template.startswith('train_vae_search_beta'):
        args.mode = 'train'

        args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'holdout'
        args.dataset_split_seed = 98765
        args.eval_set_size = 500 if args.dataset_code == 'ml-1m' else 10000

        args.dataloader_code = 'ae'
        batch = 128 if args.dataset_code == 'ml-1m' else 512
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.trainer_code = 'vae'
        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 1e-3
        args.enable_lr_schedule = False
        args.weight_decay = 0.01
        args.num_epochs = 100 if args.dataset_code == 'ml-1m' else 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.best_metric = 'NDCG@10'
        args.total_anneal_steps = 3000 if args.dataset_code == 'ml-1m' else 20000
        args.find_best_beta = True

        args.model_code = 'vae'
        args.model_init_seed = 0
        args.vae_num_hidden = 2
        args.vae_hidden_dim = 600
        args.vae_latent_dim = 200
        args.vae_dropout = 0.5

    elif args.template.startswith('train_vae_give_beta'):
        args.mode = 'train'

        args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'holdout'
        args.dataset_split_seed = 98765
        args.eval_set_size = 500 if args.dataset_code == 'ml-1m' else 10000

        args.dataloader_code = 'ae'
        batch = 128 if args.dataset_code == 'ml-1m' else 512
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.trainer_code = 'vae'
        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 1e-3
        args.enable_lr_schedule = False
        args.weight_decay = 0.01
        args.num_epochs = 100 if args.dataset_code == 'ml-1m' else 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.best_metric = 'NDCG@100'
        args.find_best_beta = False
        args.anneal_cap = 0.342
        args.total_anneal_steps = 3000 if args.dataset_code == 'ml-1m' else 20000

        args.model_code = 'vae'
        args.model_init_seed = 0
        args.vae_num_hidden = 2
        args.vae_hidden_dim = 600
        args.vae_latent_dim = 200
        args.vae_dropout = 0.5
