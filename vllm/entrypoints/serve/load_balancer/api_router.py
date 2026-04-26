# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


class LoadBalancerWeightsUpdateRequest(BaseModel):
    expert_affinity_routing_weight: float | None = Field(default=None, ge=0.0)
    kv_block_prefix_routing_weight: float | None = Field(default=None, ge=0.0)
    load_score_routing_weight: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_non_empty(self) -> "LoadBalancerWeightsUpdateRequest":
        if (
            self.expert_affinity_routing_weight is None
            and self.kv_block_prefix_routing_weight is None
            and self.load_score_routing_weight is None
        ):
            raise ValueError("At least one weight must be provided")
        return self


def _engine_client(raw_request: Request):
    return raw_request.app.state.engine_client


@router.get("/load_balancer/weights")
async def get_load_balancer_weights(raw_request: Request):
    client = _engine_client(raw_request)
    getter = getattr(client, "get_runtime_load_balancer_weights", None)
    if getter is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(
                "Runtime load-balancer weight control is unsupported for this "
                "serving topology."
            ),
        )
    try:
        state = await getter()
    except ValueError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(exc),
        ) from exc
    return JSONResponse(content=state)


@router.post("/load_balancer/weights")
async def update_load_balancer_weights(
    raw_request: Request,
    body: LoadBalancerWeightsUpdateRequest,
):
    client = _engine_client(raw_request)
    updater = getattr(client, "update_runtime_load_balancer_weights", None)
    if updater is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(
                "Runtime load-balancer weight control is unsupported for this "
                "serving topology."
            ),
        )

    try:
        state = await updater(
            expert_affinity_routing_weight=body.expert_affinity_routing_weight,
            kv_block_prefix_routing_weight=body.kv_block_prefix_routing_weight,
            load_score_routing_weight=body.load_score_routing_weight,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(exc),
        ) from exc

    logger.info("Updated runtime load-balancer weights: %s", state)
    return JSONResponse(content=state)


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return
    app.include_router(router)
