#!/bin/sh
NEW_VOCAB=$1
NEW_EMB_PREFIX=$2

python src/reinitialize_embeddings.py \
    --old_model_path xlm-roberta-base \
    --new_vocab_file tokenizers/$NEW_VOCAB \
    --embedding_output_path tokenizers/${NEW_EMB_PREFIX}_emb_32k_script.pt \
    --reinit_by_script

python src/reinitialize_embeddings.py \
    --old_model_path xlm-roberta-base \
    --new_vocab_file tokenizers/$NEW_VOCAB \
    --embedding_output_path tokenizers/${NEW_EMB_PREFIX}_emb_32k_identity.pt \
    --reinit_by_identity

python src/reinitialize_embeddings.py \
    --old_model_path xlm-roberta-base \
    --new_vocab_file tokenizers/$NEW_VOCAB \
    --embedding_output_path tokenizers/${NEW_EMB_PREFIX}_emb_32k_script+identity.pt \
    --reinit_by_script --reinit_by_identity

python src/reinitialize_embeddings.py \
    --old_model_path xlm-roberta-base \
    --new_vocab_file tokenizers/$NEW_VOCAB \
    --embedding_output_path tokenizers/${NEW_EMB_PREFIX}_emb_32k_script+position.pt \
    --reinit_by_script --reinit_by_position

python src/reinitialize_embeddings.py \
    --old_model_path xlm-roberta-base \
    --new_vocab_file tokenizers/$NEW_VOCAB \
    --embedding_output_path tokenizers/${NEW_EMB_PREFIX}_emb_32k_script+pos+ident.pt \
    --reinit_by_script --reinit_by_position --reinit_by_identity

