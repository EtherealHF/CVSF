import os

def set_default_config(cfg):
    if 'train' not in cfg:
        cfg['train'] = {}
    
    ##### default #####
    if 'output' not in cfg:
        cfg['output'] = 'runs'
    if 'tag' not in cfg:
        cfg['tag'] = 'debug'
    
    ##### data #####
    if 'persistent_workers' not in cfg['data']:
        cfg['data']['persistent_workers'] = False
    if 'train' in cfg['data'] and 'repeat' not in cfg['data']['train']:
        cfg['data']['train']['repeat'] = 1
    # augmentation 
    if 'transpose' not in cfg['data']['process']:
        cfg['data']['process']['transpose'] = False
    if 'h_flip' not in cfg['data']['process']:
        cfg['data']['process']['h_flip'] = True
    if 'v_flip' not in cfg['data']['process']:
        cfg['data']['process']['v_flip'] = True
    if 'rotation' not in cfg['data']['process']:
        cfg['data']['process']['rotation'] = False
        
    ##### train #####
    if 'auto_resume' not in cfg['train']:
        cfg['train']['auto_resume'] = False
        
    ##### test #####
    if 'test' in cfg:
        if 'round' not in cfg['test']:
            cfg['test']['round'] = False
        if 'save_image' not in cfg['test']:
            cfg['test']['save_image'] = False
        
    cfg['output'] = os.path.join(cfg.get('output', 'runs'), cfg['name'], cfg.get('tag', ''))

