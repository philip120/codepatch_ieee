from transformers import AutoTokenizer
import torch
from typing import List, Union
import os
import sys

from modeling_codepatch import CodeEncoderConfig
from ast_parser.matlab_parser import MatlabParser
from ast_parser.matlab_lexer import MatlabLexer
from ast_parser.ast_utils import get_semantic_patches


class CodePatchProcessor:
    def __init__(self, code_tokenizer_name: str, text_tokenizer_name: str, config: CodeEncoderConfig, device: str = None):
        self.code_tokenizer = AutoTokenizer.from_pretrained(code_tokenizer_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name)
        self.config = config
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Suppress the SLY parsing errors from filling up the console
        self.parser = MatlabParser()
        # self.parser.error_output = open(os.devnull, 'w')  # Temporarily disabled for debugging
        
        self.lexer = MatlabLexer()
        # self.lexer.error_output = open(os.devnull, 'w')  # Temporarily disabled for debugging

    def _process_code(self, code_strings: List[str]):
        """ Processes a batch of code strings using AST-based semantic patching. """
        all_patches_tensors = []
        all_masks_tensors = []

        for code_string in code_strings:
            cleaned_code = code_string.replace('\\n', ' ').replace('\n', ' ')
            semantic_patches = get_semantic_patches(cleaned_code, self.parser, self.lexer)

            print(f"Num semantic patches: {len(semantic_patches)}")  # Debug print

            if not semantic_patches:
                semantic_patches = [""]

            inputs = self.code_tokenizer(
                semantic_patches,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.config.patch_length,
            )
            
            tokenized_patches = inputs["input_ids"]
            
            num_real_patches = tokenized_patches.shape[0]
            num_padding_patches = self.config.num_patches - num_real_patches

            if num_padding_patches > 0:
                padding = torch.full(
                    (num_padding_patches, self.config.patch_length), 
                    self.code_tokenizer.pad_token_id, 
                    dtype=torch.long
                )
                final_patches = torch.cat((tokenized_patches, padding), dim=0)
            else:
                final_patches = tokenized_patches[:self.config.num_patches]

            attention_mask = (final_patches != self.code_tokenizer.pad_token_id).long()

            all_patches_tensors.append(final_patches)
            all_masks_tensors.append(attention_mask)
        
        batch_input_ids = torch.stack(all_patches_tensors).to(self.device)
        batch_attention_mask = torch.stack(all_masks_tensors).to(self.device)

        return batch_input_ids, batch_attention_mask

    def _process_text(self, text_prompts: List[str]):
        """ Processes a batch of text prompts into tokenized tensors. """
        prompt_inputs = self.text_tokenizer(
            text_prompts, return_tensors="pt", padding="longest", truncation=True, max_length=512
        )
        prompt_input_ids = prompt_inputs["input_ids"].to(self.device)
        prompt_attention_mask = prompt_inputs["attention_mask"].to(self.device)
        return prompt_input_ids, prompt_attention_mask

    def __call__(self, code: Union[str, List[str]], text: Union[str, List[str]]):
        if isinstance(code, str):
            code = [code]
        if isinstance(text, str):
            text = [text]

        code_input_ids, code_attention_mask = self._process_code(code)
        prompt_input_ids, prompt_attention_mask = self._process_text(text)

        return {
            "code_input_ids": code_input_ids,
            "code_attention_mask": code_attention_mask,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
        } 