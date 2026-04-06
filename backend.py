import numpy as np
import scipy.stats as s
import pickle
import config
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import Query
from typing import Annotated


def determine_placement_posterior_probability(input_features):

    input_features = np.array(input_features).reshape(1, -1)

    eig_vectors = np.load("eigen_vectors.npy")
    new_input_features = np.matmul(input_features, eig_vectors)

    log_likelihood_0 = 0.0
    log_likelihood_1 = 0.0

    with open("likelihood_distribution_params.pkl", "rb") as file_handle:
        likelihood_distribution_params = pickle.load(file_handle)

    for input_feat, input_feat_value in zip(config.INPUT_FEATURE_NAMES, new_input_features[0]):
        mu_0, sigma_0 = likelihood_distribution_params[0][input_feat]
        mu_1, sigma_1 = likelihood_distribution_params[1][input_feat]

        p0 = s.norm.pdf(input_feat_value, mu_0, sigma_0)
        p1 = s.norm.pdf(input_feat_value, mu_1, sigma_1)

        p0 = max(p0, 1e-10)
        p1 = max(p1, 1e-10)

        log_likelihood_0 += np.log(p0)
        log_likelihood_1 += np.log(p1)

    log_posterior_0 = log_likelihood_0 + np.log(1 - config.PLACEMENT_EQUALS_1_PRIOR)
    log_posterior_1 = log_likelihood_1 + np.log(config.PLACEMENT_EQUALS_1_PRIOR)

    max_log = max(log_posterior_0, log_posterior_1)

    p0 = np.exp(log_posterior_0 - max_log)
    p1 = np.exp(log_posterior_1 - max_log)

    prob_1 = p1 / (p0 + p1)

    if prob_1 > 0.5:
        result_msg = f"You are likely to get placed with probability {round(prob_1, 3)}"
    else:
        result_msg = f"Placement chances are low with probability {round(prob_1, 3)}"

    return {"probability": round(prob_1, 3), "result": result_msg}


app = FastAPI()


class InputFeatureVector(BaseModel):
    iq: Annotated[int, Query(ge=40, le=160)]
    previous_semester_result: Annotated[float, Query(ge=0, le=10)]
    cgpa: Annotated[float, Query(ge=0, le=10)]
    communication_skills: Annotated[int, Query(ge=0, le=10)]
    projects_completed: Annotated[int, Query(ge=0, le=10)]


@app.get("/")
def home_page():
    return {
        "message": "This web App predicts the Probability of a Student being Placed"
    }


@app.post("/compute-probability")
def compute_probability(input_features: InputFeatureVector):

    input_feature_values_list = [
        input_features.iq,
        input_features.previous_semester_result,
        input_features.cgpa,
        input_features.communication_skills,
        input_features.projects_completed,
    ]

    response = determine_placement_posterior_probability(input_feature_values_list)

    return response