# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


class EplbStepIntervalUpdateRequest(BaseModel):
    step_interval: int = Field(ge=1)


class PlacementRoutingDumpUpdateRequest(BaseModel):
    dump_dir: str | None = None


class MoEDispatchTrafficDumpUpdateRequest(BaseModel):
    dump_dir: str | None = None


def _engine_client(raw_request: Request):
    return raw_request.app.state.engine_client


@router.get("/eplb/step_interval")
async def get_eplb_step_interval(raw_request: Request):
    client = _engine_client(raw_request)
    getter = getattr(client, "get_runtime_eplb_step_interval", None)
    if getter is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(
                "Runtime EPLB step_interval control is unsupported for this "
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


@router.post("/eplb/step_interval")
async def update_eplb_step_interval(
    raw_request: Request,
    body: EplbStepIntervalUpdateRequest,
):
    client = _engine_client(raw_request)
    updater = getattr(client, "update_runtime_eplb_step_interval", None)
    if updater is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(
                "Runtime EPLB step_interval control is unsupported for this "
                "serving topology."
            ),
        )

    try:
        state = await updater(step_interval=body.step_interval)
    except ValueError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(exc),
        ) from exc

    logger.info("Updated runtime EPLB step_interval: %s", state)
    return JSONResponse(content=state)


@router.get("/eplb/placement_routing_dump")
async def get_placement_routing_dump(raw_request: Request):
    client = _engine_client(raw_request)
    getter = getattr(client, "get_runtime_placement_routing_dump", None)
    if getter is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(
                "Runtime placement routing dump control is unsupported for "
                "this serving topology."
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


@router.post("/eplb/placement_routing_dump")
async def update_placement_routing_dump(
    raw_request: Request,
    body: PlacementRoutingDumpUpdateRequest,
):
    client = _engine_client(raw_request)
    updater = getattr(client, "update_runtime_placement_routing_dump", None)
    if updater is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(
                "Runtime placement routing dump control is unsupported for "
                "this serving topology."
            ),
        )

    try:
        state = await updater(dump_dir=body.dump_dir)
    except ValueError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(exc),
        ) from exc

    logger.info("Updated runtime placement routing dump state: %s", state)
    return JSONResponse(content=state)


@router.get("/eplb/moe_dispatch_traffic_dump")
async def get_moe_dispatch_traffic_dump(raw_request: Request):
    client = _engine_client(raw_request)
    getter = getattr(client, "get_runtime_moe_dispatch_traffic_dump", None)
    if getter is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(
                "Runtime MoE dispatch traffic dump control is unsupported for "
                "this serving topology."
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


@router.post("/eplb/moe_dispatch_traffic_dump")
async def update_moe_dispatch_traffic_dump(
    raw_request: Request,
    body: MoEDispatchTrafficDumpUpdateRequest,
):
    client = _engine_client(raw_request)
    updater = getattr(client, "update_runtime_moe_dispatch_traffic_dump", None)
    if updater is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(
                "Runtime MoE dispatch traffic dump control is unsupported for "
                "this serving topology."
            ),
        )

    try:
        state = await updater(dump_dir=body.dump_dir)
    except ValueError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(exc),
        ) from exc

    logger.info("Updated runtime MoE dispatch traffic dump state: %s", state)
    return JSONResponse(content=state)


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return
    app.include_router(router)
