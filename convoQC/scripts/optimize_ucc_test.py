from .optimize_ucc import optimize_ucc
# pylint: ignore unused-import


def optimize_ucc_test():
    # just check it runs...
    optimize_ucc('H,0,0,0;H,1,0,0;H,0,1,0;H,0,0,1', maxiter=2)
