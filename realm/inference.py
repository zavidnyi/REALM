import time
import numpy as np
from PIL import Image
from openpi_client import websocket_client_policy, image_tools
from websockets.exceptions import WebSocketException
import msgpack
import omnigibson as og

_RECONNECT_DELAY_S = 5.0
_MAX_RECONNECT_ATTEMPTS = 20


def extract_proprio_from_obs(obs: dict) -> tuple[np.ndarray, float]:
    proprio = obs['franka']['proprio'].cpu().numpy()
    robot_state = proprio[:7]
    gripper_state = proprio[7] / 0.05
    return robot_state, gripper_state


def extract_images_from_obs(obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_im = obs['external']['external_sensor0']['rgb'].cpu().numpy()[..., :3]
    base_im_second = obs['external']['external_sensor1']['rgb'].cpu().numpy()[..., :3]
    wrist_im = obs['franka']['franka:gripper_link_camera:Camera:0']['rgb'].cpu().numpy()[..., :3]
    return base_im, base_im_second, wrist_im


def extract_from_obs(obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    robot_state, gripper_state = extract_proprio_from_obs(obs)
    base_im, base_im_second, wrist_im = extract_images_from_obs(obs)
    return base_im, base_im_second, wrist_im, robot_state, gripper_state


class InferenceClient:
    def __init__(self, model_type: str, port: int, host: str = "localhost") -> None:
        self.model_type = model_type
        self.host = host
        self.port = port
        self.client: websocket_client_policy.WebsocketClientPolicy | None = None
        if model_type != "debug":
            self._connect()

    def _connect(self) -> None:
        for attempt in range(1, _MAX_RECONNECT_ATTEMPTS + 1):
            try:
                og.log.info(f"Connecting to server (attempt {attempt}/{_MAX_RECONNECT_ATTEMPTS})...")
                self.client = websocket_client_policy.WebsocketClientPolicy(
                    host=self.host,
                    port=self.port,
                )
                og.log.info("Connected!")
                return
            except Exception as exc:
                og.log.warning(f"Connection failed: {exc}. Retrying in {_RECONNECT_DELAY_S}s...")
                time.sleep(_RECONNECT_DELAY_S)
        raise RuntimeError(
            f"Could not connect to policy server at {self.host}:{self.port} after {_MAX_RECONNECT_ATTEMPTS} attempts"
        )

    def _call_with_reconnect(self, obs_dict: dict) -> dict:
        for attempt in range(1, _MAX_RECONNECT_ATTEMPTS + 1):
            try:
                assert self.client is not None
                return self.client.infer(obs_dict)
            except (WebSocketException, OSError, msgpack.UnpackException) as exc:
                og.log.warning(f"Inference error (attempt {attempt}): {exc}. Reconnecting...")
                self._connect()
        raise RuntimeError("Failed to get inference response after reconnection attempts")

    def infer(
        self,
        instruction: str,
        base_im: np.ndarray,
        base_im_second: np.ndarray,
        wrist_im: np.ndarray,
        robot_state: np.ndarray,
        gripper_state: float,
        use_base_im_second: bool = False,
    ) -> np.ndarray:
        if self.model_type == "debug":
            return np.atleast_1d(np.zeros(8))

        if self.model_type == "GR00T":
            base_im_resized = np.asarray(Image.fromarray(base_im).resize((320, 180))).astype(np.uint8)
            base_im_second_resized = np.asarray(Image.fromarray(base_im_second).resize((320, 180))).astype(np.uint8)
            wrist_im_resized = np.asarray(Image.fromarray(wrist_im).resize((320, 180))).astype(np.uint8)

            obs_dict = {
                "prompt": [instruction],
                "state.joint_position": np.array(robot_state).astype(np.float32).reshape(1, 7),
                "state.gripper_position": np.atleast_1d(np.array(gripper_state)).astype(np.float32).reshape(1, 1),
                "video.exterior_image_1": base_im_resized[None],
                "video.exterior_image_2": base_im_second_resized[None],
                "video.wrist_image": wrist_im_resized[None],
            }
            pred = self._call_with_reconnect(obs_dict)
            return np.concatenate(
                [pred["action.joint_position"], pred["action.gripper_position"].reshape(-1, 1)],
                axis=-1,
            )

        img_to_use = base_im_second if use_base_im_second else base_im
        obs_dict = {
            "prompt": instruction,
            "observation/joint_position": robot_state,
            "observation/gripper_position": np.atleast_1d(np.array(gripper_state)),
            "observation/exterior_image_1_left": image_tools.resize_with_pad(img_to_use, 224, 224),
            "observation/wrist_image_left": image_tools.resize_with_pad(wrist_im, 224, 224),
        }
        pred = self._call_with_reconnect(obs_dict)
        return pred["actions"]
