from fairseq.tasks.multitask_translation_all_lgid import MultitaskTranslationTaskAllLGID, MultiTaskTranslationConfig
from .generator_with_reencoder import SequenceGeneratorWithReencoder
from fairseq.tasks import LegacyFairseqTask, register_task


@register_task("multitasktranslationalllgidreecndoer", dataclass=MultiTaskTranslationConfig)
class MultitaskTranslationTaskAllLGIDReencoder(MultitaskTranslationTaskAllLGID):
    
    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        seq_gen_cls=SequenceGeneratorWithReencoder
        return self._build_generator(models, args, seq_gen_cls, extra_gen_cls_kwargs)
