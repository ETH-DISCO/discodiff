import os
import json
import torch
import argparse
import torchaudio
from encodec.utils import convert_audio
from transformers import AutoProcessor, ClapModel

import re

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'
def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get CLAP audio encoder
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    print("CLAP model loaded")

    if args.text_json_path is not None:
        with open(args.text_json_path) as f:
            text_feat_metadata = json.load(f)
    elif args.text_dir is not None:
        text_files = os.listdir(args.text_dir)
        text_feat_metadata = {}
        for filename in text_files:
            if len(filename)>4 and filename[-4:] == ".txt":
                audio_name = os.path.splitext(filename)[0].split("_")[0]
                text_file = os.path.join(args.text_dir, filename)
                with open(text_file, "r") as file:
                    caption = file.readline().strip()
                text_feat_metadata[audio_name] = {'text': caption}
    else:
        assert 0, "Should provide caption json file or caption directory of txt files."

    cosine_sim_list = []
    audio_files = os.listdir(args.audio_dir)
    with torch.no_grad():
        for audio_file in audio_files:
            audio_name = os.path.splitext(audio_file)[0].split("_")[0]
            audio_path = os.path.join(args.audio_dir, audio_file)
            if audio_name in text_feat_metadata:
                wav, sr = torchaudio.load(audio_path)
                wav = convert_audio(wav, sr, 48000, 1).squeeze(0)

                input_text = text_feat_metadata[audio_name]['text']
                if args.max_num_sentences is not None:
                    input_text_sentences = split_into_sentences(input_text)
                    if len(input_text_sentences) > args.max_num_sentences:
                        input_text_sentences = input_text_sentences[:args.max_num_sentences]
                    input_text = " ".join(input_text_sentences)

                input_text = [input_text]
                inputs = processor(text=input_text, audios=wav, return_tensors="pt", sampling_rate=48000, padding=True)
                outputs = model(**inputs)
                similarity = torch.cosine_similarity(
                    outputs.text_embeds,
                    outputs.audio_embeds,
                    dim=1,
                    eps=1e-12,
                ).item()
                # similarity = model(**inputs).logits_per_audio
                cosine_sim_list.append(similarity)
                print(audio_name, ":", similarity, "; average:", sum(cosine_sim_list) / len(cosine_sim_list))

    print("---final clap score---")
    print(sum(cosine_sim_list) / len(cosine_sim_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in training DAC latent diffusion.')
    parser.add_argument(
        '-audio-dir', type=str, default='',
        help='the directory that model results are saved'
    )
    parser.add_argument(
        '-text-dir', type=str, nargs="?",
        help='the directory that captions are saved'
    )
    parser.add_argument(
        '-text-json-path', type=str, nargs="?",
        help='the json file that captions are saved'
    )
    parser.add_argument(
        '--max-num-sentences', type=int, nargs="?",
        help='if text prompt is too long, prune it to shorter sentences'
    )
    args = parser.parse_args()
    main(args)