"""
Contains the response generation modules of the agent.
"""

import json
import os

from Module.prompts import build_generator_prompts, format_patient_simulator_prompts
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction, DefaultEmbeddingFunction

class Generator():
    def __init__(self, client, sampling_dataset, sample_id, logger, args):
        self.args = args
        self.logger = logger
        self.embedding_function = DefaultEmbeddingFunction()
        self.data_loader = ImageLoader()

        self.LLMclient = client

        self.usr_name = args.usr_name
        self.agent_name = args.agent_name

        self.sampling_dataset = sampling_dataset
        self.sampling_step = args.sampling_step
        self.sampling_file = os.path.join(args.sampling_path, args.sampling_file_name)


    def select_prompts(self, inquiry, context, memories, user_traits, agent_traits,
                       therapist_action=None):
        return build_generator_prompts(
            self.usr_name,
            self.agent_name,
            inquiry,
            context,
            memories,
            user_traits,
            agent_traits,
            therapist_action=therapist_action,
        )

    def sampling(self, inquiry, reference, context, memories, user_traits, agent_traits):
        sys_prompt, user_prompt = self.select_prompts(inquiry, context, memories, user_traits, agent_traits)

        prompted_item = {
            "conversations": [
                {
                    "role": "system",
                    "content": sys_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
                {
                    "role": "assistant",
                    "content": reference
                }
            ]
        }
        self.sampling_dataset.append(prompted_item)
        if len(self.sampling_dataset) % self.sampling_step == 0:
            with open(self.sampling_file, "w") as f:
                json.dump(self.sampling_dataset, f, indent=4)

            self.logger.info(prompted_item)

        return " "

    def response_build(self, inquiry, context, memories, user_traits, agent_traits,
                       therapist_action=None):
        sys_prompt, user_prompt = self.select_prompts(
            inquiry, context, memories, user_traits, agent_traits,
            therapist_action=therapist_action,
        )

        response = self.LLMclient.employ(sys_prompt, user_prompt, "ResponseGenerator")

        return response

    def select_patient_prompts(
        self, doctor_reply, context, memories, state_after_action=""
    ):
        return format_patient_simulator_prompts(
            self.usr_name,
            self.agent_name,
            doctor_reply,
            context,
            memories,
            state_after_action=state_after_action,
        )

    def patient_response_build(
        self, doctor_reply, context, memories, state_after_action=""
    ):
        sys_prompt, user_prompt = self.select_patient_prompts(
            doctor_reply, context, memories, state_after_action
        )
        raw = self.LLMclient.employ(sys_prompt, user_prompt, "ResponseGenerator")
        return self._strip_patient_response(raw)

    def patient_sampling(
        self,
        doctor_reply,
        reference_next_patient,
        context,
        memories,
        state_after_action="",
    ):
        sys_prompt, user_prompt = self.select_patient_prompts(
            doctor_reply, context, memories, state_after_action
        )
        prompted_item = {
            "conversations": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": reference_next_patient},
            ]
        }
        self.sampling_dataset.append(prompted_item)
        if len(self.sampling_dataset) % self.sampling_step == 0:
            with open(self.sampling_file, "w") as f:
                json.dump(self.sampling_dataset, f, indent=4)
            self.logger.info(prompted_item)
        return " "

    @staticmethod
    def _strip_patient_response(text: str) -> str:
        if not text:
            return ""
        s = text.strip()
        key = "PATIENT_RESPONSE:"
        if key in s:
            s = s.split(key, 1)[-1].strip()
        return s.strip().strip('"').strip("'")
