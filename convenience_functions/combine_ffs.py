"""Functionality for combining offxml files into a single offxml file."""

from pathlib import Path

import loguru
from openff.toolkit import ForceField

logger = loguru.logger


def combine_force_fields(
    ff_to_combine_paths: dict[str, Path],
    output_file: Path,
    base_ff: str = "openff_unconstrained-2.3.0.offxml",
) -> ForceField:
    """
    Combines multiple OpenFF force field XML files into a single force field, starting with a base force field.

    Parameters:
        ff_to_combine_paths (dict[str, Path]): A dictionary mapping force field names to their file paths.
        output_file (Path): Path to the output XML file for the combined force field.
        base_ff (str): Path to the base force field XML file.

    Returns:
        ForceField: The combined force field.
    """
    # Load the base force field
    original_force_field = ForceField(base_ff)
    combined_force_field = ForceField(base_ff)

    # Load the actual force field objects from the file paths
    ff_to_combine = {
        ff_name: ForceField(str(ff_path))
        for ff_name, ff_path in ff_to_combine_paths.items()
    }

    for ff_name, ff in ff_to_combine.items():
        # Load each force field and add its parameters to the combined force field
        for handler_name in ff.registered_parameter_handlers:
            handler = ff.get_parameter_handler(handler_name)
            combined_handler = combined_force_field.get_parameter_handler(handler_name)
            original_handler = original_force_field.get_parameter_handler(handler_name)
            existing_parameter_smirks = {
                param.smirks for param in combined_handler.parameters
            }
            original_parameter_smirks = {
                param.smirks for param in original_handler.parameters
            }
            new_parameters = existing_parameter_smirks - original_parameter_smirks

            for parameter in handler.parameters:
                # Skip constraints
                if handler_name == "Constraints":
                    continue

                # Make the parameter id unique by adding the input file directory name
                parameter.id += f"_{ff_name}"

                # Skip parameters that are already included in the base force field
                if parameter.smirks in original_parameter_smirks:
                    # # Make sure the parameter is identical to the one in the base force field
                    # original_param = original_handler[parameter.smirks]
                    # if not all(
                    #     parameter.to_dict()[key] == original_param.to_dict()[key]
                    #     for key in parameter.to_dict()
                    #     if key != "id"
                    # ):
                    #     logger.error(
                    #         f"Parameter {parameter.smirks} from {ff_name} is not identical to the parameter in the base force field."
                    #     )
                    #     raise ValueError(
                    #         f"Parameter {parameter.smirks} from {ff_name} is not identical to the parameter in the base force field."
                    #     )
                    continue

                # Raise an error if a parameter is already present in the combined force field
                if parameter.smirks in new_parameters:
                    current_new_params = combined_handler[parameter.smirks]
                    # Check if the dicts (other than id) are the same
                    if all(
                        parameter.to_dict()[key] == current_new_params.to_dict()[key]
                        for key in parameter.to_dict()
                        if key != "id"
                    ):
                        logger.info(
                            f"Parameter {parameter.smirks} from {ff_name} is identical to existing parameter. Skipping."
                        )
                        continue
                    logger.error(
                        f"New parameter {parameter} is not identical to existing parameter with the same SMIRKS: {current_new_params}"
                    )
                    continue
                    # raise ValueError(
                    #     f"New parameter ID {parameter.id} {parameter} already exists in the combined force field."
                    # )

                combined_handler.add_parameter(parameter.to_dict())

    # Save the combined force field to the output file
    combined_force_field.to_file(output_file)
    logger.info(f"Combined force field saved to {output_file}")

    return combined_force_field
