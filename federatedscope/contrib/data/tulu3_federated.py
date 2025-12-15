from federatedscope.llm.dataset.tulu3_federated import load_tulu3_federated_data
from federatedscope.register import register_data


def call_tulu3_federated_data(config, client_cfgs):
    if config.data.type.lower() == "tulu3_federated":
        return load_tulu3_federated_data(config, client_cfgs)


register_data("tulu3_federated", call_tulu3_federated_data)
