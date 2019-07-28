

import pandas


class VarTransform:
    """build a treatment plan for a numeric outcome (regression)"""

    def __init__(self, incoming_column_name, derived_column_names, treatment):
        self.incoming_column_name_ = incoming_column_name
        self.derived_column_names_ = derived_column_names.copy()
        self.treatment_ = treatment
        self.need_cross_treatment_ = False
        self.refitter_ = None

    def transform(self, data_frame):
        raise Exception("base method called")


class MappedCodeTransform(VarTransform):
    def __init__(self, incoming_column_name, derived_column_name, treatment, code_book):
        VarTransform.__init__(
            self, incoming_column_name, [derived_column_name], treatment
        )
        self.code_book_ = code_book

    def transform(self, data_frame):
        incoming_column_name = self.incoming_column_name_
        derived_column_name = self.derived_column_names_[0]
        sf = pandas.DataFrame({incoming_column_name: data_frame[incoming_column_name]})
        na_posns = sf[incoming_column_name].isnull()
        sf.loc[na_posns, incoming_column_name] = "_NA_"
        res = pandas.merge(
            sf, self.code_book_, on=[self.incoming_column_name_], how="left", sort=False
        )  # ordered by left table rows
        res = res[[derived_column_name]].copy()
        res.loc[res[derived_column_name].isnull(), derived_column_name] = 0
        return res


class YAwareMappedCodeTransform(MappedCodeTransform):
    def __init__(
            self,
            incoming_column_name,
            derived_column_name,
            treatment,
            code_book,
            refitter,
            extra_args,
            params,
    ):
        MappedCodeTransform.__init__(
            self,
            incoming_column_name=incoming_column_name,
            derived_column_name=derived_column_name,
            treatment=treatment,
            code_book=code_book,
        )
        self.need_cross_treatment_ = True
        self.refitter_ = refitter
        self.extra_args_ = extra_args
        self.params_ = params
