from iglm import IgLM
import math
# from COLD_decoding import cold_decoding

iglm = IgLM()


### Generation example
# prompt_sequence = "QVQ"
# chain_token = "[HEAVY]"
# species_token = "[CAMEL]"
# num_seqs = 10

# generated_seqs = iglm.generate(
#     chain_token,
#     species_token,
#     prompt_sequence=prompt_sequence,
#     num_to_generate=num_seqs,
# )


sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKEYYMHWVRQAPGKGLEWVGLIDPEQGNTIYDPKFQDRATISADNSKNTAYLQMNSLRAEDTAVYYCARDTAAYFDYWGQGTLVTVS"
# sequence = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
#sequence = "EV"

print("sequence length",len(sequence))
chain_token = "[HEAVY]"
species_token = "[HUMAN]"

sequence_logits, log_likelihood = iglm.likelihood_optimization( # chain,species,...,sep 
    sequence,
    chain_token,
    species_token
)

print(sequence_logits)
# out = cold_decoding.decode(sequence_logits)

#Higher log-likelihood is better? string of all 120 ish As = -3.6, string of actual generated Ig = -0.1
