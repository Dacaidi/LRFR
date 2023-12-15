from importlib import import_module


def factory(subdir, module_name, func):
    module = import_module(
        '.' + module_name, package=subdir
    )
    model = getattr(module, func)
    return model


def count_parameter(model):
    clf_param_num = sum(p.numel() for p in model.parameters())
    return clf_param_num


def find_module_name(module, root):
    for name, m in root.named_modules():
        if m == module:
            return name
    return None


def find_module_by_name(root, name):
    for n, m in root.named_modules():
        if n == name:
            return m
    return None


def pruned_weight_to_original_model_conv(module_pruned, module_origin, idx0, idx1):
    for i in range(len(idx0)):
        for j in range(len(idx1)):
            module_origin.weight.data[idx1[j], idx0[i], :, :] = module_pruned.weight.data[j, i, :, :].clone()
