def register_all_backends():
    from .mhlo import _dispatch_mhlo
    from .tf import _dispatch_tf
