'''
Based on code used for the experiments conducted in the submitted paper 
"Fixed-Weight Difference Target Propagation" by K. K. S. Wu, K. C. K. Lee, and T. Poggio.

Adaptation for Continual Learning by emmagg6.

'''


def set_params(kwargs):
    name = {"ff1": "forward_function_1",
            "ff2": "forward_function_2",
            "bf1": "backward_function_1",
            "bf2": "backward_function_2"}
    params = {}
    sparse_ratio = ("-sparse-" + str(kwargs["sparse_ratio"])
                    ) if 1 >= kwargs["sparse_ratio"] >= 0 else ""

    params["last"] = "linear"
    if kwargs["algorithm"] == "TP":
        params["ff1"] = {"type": "identity",
                         "init": None,
                         "act": "linear"}
        params["ff2"] = {"type": "parameterized",
                         "init": kwargs[name["ff2"] + "_init"],
                         "act": "tanh"}
        params["bf1"] = {"type": "parameterized",
                         "init": kwargs[name["bf1"] + "_init"],
                         "act": "tanh"}
        params["bf2"] = {"type": "identity",
                         "init": None,
                         "act": "linear"}
    elif kwargs["algorithm"] == "DTP":
        params["ff1"] = {"type": "identity",
                         "init": None,
                         "act": "linear"}
        params["ff2"] = {"type": "parameterized",
                         "init": kwargs[name["ff2"] + "_init"],
                         "act": "tanh"}
        params["bf1"] = {"type": "parameterized",
                         "init": kwargs[name["bf1"] + "_init"],
                         "act": "tanh"}
        params["bf2"] = {"type": "difference",
                         "init": None,
                         "act": "linear"}
    elif kwargs["algorithm"] == "DTP-BN":
        params["ff1"] = {"type": "identity",
                         "init": None,
                         "act": "linear"}
        params["ff2"] = {"type": "parameterized",
                         "init": kwargs[name["ff2"] + "_init"],
                         "act": "tanh-BN"}
        params["bf1"] = {"type": "parameterized",
                         "init": kwargs[name["bf1"] + "_init"],
                         "act": "tanh-BN"}
        params["bf2"] = {"type": "difference",
                         "init": None,
                         "act": "linear-BN"}
    elif kwargs["algorithm"] == "FWDTP":
        params["ff1"] = {"type": "identity",
                         "init": None,
                         "act": "linear"}
        params["ff2"] = {"type": "parameterized",
                         "init": kwargs[name["ff2"] + "_init"],
                         "act": "tanh"}
        params["bf1"] = {"type": "parameterized",
                         "init": kwargs[name["bf1"] + "_init"] + sparse_ratio,
                         "act": "tanh"}
        params["bf2"] = {"type": "difference",
                         "init": None,
                         "act": "linear"}
    elif kwargs["algorithm"] == "FWDTP-BN":
        params["ff1"] = {"type": "parameterized",
                         "init": None,
                         "act": "linear-BN"}
        params["ff2"] = {"type": "parameterized",
                         "init": kwargs[name["ff2"] + "_init"],
                         "act": "tanh-BN"}
        params["bf1"] = {"type": "parameterized",
                         "init": kwargs[name["bf1"] + "_init"] + sparse_ratio,
                         "act": "tanh-BN"}
        params["bf2"] = {"type": "difference",
                         "init": None,
                         "act": "linear-BN"}
        params["last"] = kwargs["forward_last_activation"]
    elif kwargs["algorithm"] == "ITP":
        params["ff1"] = {"type": "identity",
                         "init": None,
                         "act": "linear"}
        params["ff2"] = {"type": "parameterized",
                         "init": kwargs[name["ff2"] + "_init"],
                         "act": "tanh"}
        params["bf1"] = {"type": "identity",
                         "init": None,
                         "act": "linear"}
        params["bf2"] = {"type": "difference",
                         "init": None,
                         "act": "linear"}
    elif kwargs["algorithm"] == "ITP-BN":
        params["ff1"] = {"type": "identity",
                         "init": None,
                         "act": "linear"}
        params["ff2"] = {"type": "parameterized",
                         "init": kwargs[name["ff2"] + "_init"],
                         "act": "tanh-BN"}
        params["bf1"] = {"type": "identity",
                         "init": None,
                         "act": "linear-BN"}
        params["bf2"] = {"type": "difference",
                         "init": None,
                         "act": "linear-BN"}
        
    print("Parameters set for " + kwargs["algorithm"] + " algorithm: \n" )
    print(params)
    return params