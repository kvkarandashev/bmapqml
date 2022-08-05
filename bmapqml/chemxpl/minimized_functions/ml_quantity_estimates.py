from ..utils import coord_info_from_tp


class RepGenFuncProblem(Exception):
    pass


class InterfacedModel:
    def __init__(self, model):
        """
        Returns prediction of the KRR model for a TrajectoryPoint object.
        """
        self.model = model

    def __call__(self, trajectory_point_in):
        try:
            representation = self.representation_func(trajectory_point_in)
            return self.model(representation)
        except RepGenFuncProblem:
            return None


class FF_based_model(InterfacedModel):
    def __init__(self, *args, num_ff_attempts=1, ff_type="MMFF", **kwargs):
        """
        Prediction of a model for TrajectoryPoint object that uses FF coordinates as input.
        num_ff_attempts : since FF might not converge the first time, specify how many attempts should be made to see whether it does converge.
        """
        super().__init__(*args, **kwargs)

        self.num_ff_attempts = num_ff_attempts
        self.ff_type = ff_type

        self.add_info_dict = {"coord_info": coord_info_from_tp}
        self.kwargs_dict = {
            "coord_info": {
                "num_attempts": self.num_ff_attempts,
                "ff_type": self.ff_type,
            }
        }

    def representation_func(self, trajectory_point_in):
        coordinates = trajectory_point_in.calc_or_lookup(
            self.add_info_dict, kwargs_dict=self.kwargs_dict
        )["coord_info"]["coordinates"]
        if coordinates is None:
            raise RepGenFuncProblem
        nuclear_charges = trajectory_point_in.egc.true_ncharges()
        return self.coord_representation_func(coordinates, nuclear_charges)
