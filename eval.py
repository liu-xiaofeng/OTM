import numpy as np

def diag_error(Or, Tr):
    oind = Or.flatten();
    tind = Tr.flatten();

    main_diag = 0.0;
    tri_diag = 0.0;
    for ix in range(len(oind)):
        o = oind[ix];
        t = tind[ix];
        if o == t:
            main_diag += 1.0;
        elif abs(o - t) == 1:
            tri_diag += 1.0;
    return (main_diag / len(oind), (main_diag + tri_diag) / len(oind));


