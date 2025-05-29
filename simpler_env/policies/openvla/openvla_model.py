import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import numpy as np
from pathlib import Path
from typing import Optional, Union, Sequence

import torch
from PIL import Image

# for local checkpoint loading
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)

import tensorflow as tf
from transforms3d.euler import euler2axangle

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

class OpenVLAInference:
    def __init__(
        self,
        openvla_path: Union[str, Path] = "openvla/openvla-7b",     # HF Hub Path (or path to local run directory),
        attn_implementation: Optional[str] = "flash_attention_2",
        action_scale: float = 1.0,
    ) -> None:
        # Register OpenVLA model to HF AutoClasses (only needed when using a local vla path)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        self.openvla_path, self.attn_implementation = openvla_path, attn_implementation
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(
            self.openvla_path, trust_remote_code=True
        )
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if os.path.isdir(self.openvla_path):
            with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
                self.vla.norm_stats = json.load(f)
                
        self.action_scale = action_scale
        
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.gripper_is_closed = False
        self.previous_gripper_action = None
                
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = tf.image.resize(
            image,
            size=(self.image_size, self.image_size),
            method="lanczos3",
            antialias=True,
        )
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        return image
                
    def reset(self, task_description: str) -> None:
        self.prompt = get_openvla_prompt(task_description, self.openvla_path)
        
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.gripper_is_closed = False
        self.previous_gripper_action = None
        
    def step(
        self, 
        image: np.ndarray, 
        task_description: Optional[str] = None, 
        *args, 
        **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if task_description is not None:
            if task_description != self.task_description:
                # task description has changed; reset the policy state
                self.reset(task_description)
                
        assert image.dtype == np.uint8
        image = self._resize_image(image)
        
        # Run VLA Inference
        inputs = self.processor(
            self.prompt, Image.fromarray(image).convert("RGB")
        ).to(self.device, dtype=torch.bfloat16)
        # raw_actions = self.vla.predict_action(
        #     **inputs, unnorm_key=unnorm_key, do_sample=False # Need a way to get the unnorm key
        # )
        raw_actions = self.vla.predict_action(
            **inputs, do_sample=False
        )
        
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }
        
        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale
        
        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]

            # alternative implementation
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                    self.previous_gripper_action - current_gripper_action
                )  # google robot 1 = close; -1 = open
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = (
                2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            )  # binarize gripper action to 1 (open) and -1 (close)
            # self.gripper_is_closed = (action['gripper'] < 0.0)

        action["terminate_episode"] = np.array([0.0])
        
        return raw_action, action

    def visualize_epoch(self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)