from .wiki2vec import Wiki2VecExtractor

EXTRACTORS = {
    Wiki2VecExtractor.code(): Wiki2VecExtractor,
}


def extractors_factory(args):
    extractors = {key: value(args) for key, value in EXTRACTORS.items()
                  if key in args.additional_inputs}
    return extractors
