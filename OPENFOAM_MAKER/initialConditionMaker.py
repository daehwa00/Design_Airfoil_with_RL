import math


def make_initial_condition(angle_of_attack, freestream_velocity):
    initial_condition_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  11
     \\\\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       volVectorField;
    object      U;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];   // [M L^-1 T^-2 Q^0 N^0 J^0]

internalField   uniform ({freestream_velocity * math.cos(math.radians(angle_of_attack))} {freestream_velocity * math.sin(math.radians(angle_of_attack))} 0);

boundaryField
{{
    inlet
    {{
        type            freestreamVelocity;
        freestreamValue $internalField;
    }}

    outlet
    {{
        type            freestreamVelocity;
        freestreamValue $internalField;
    }}

    walls
    {{
        type            noSlip;
    }}

    frontAndBack
    {{
        type            empty;
    }}
}}

// ************************************************************************* //

    """

    # blockMeshDict 파일 생성
    with open("./U", "w") as f:
        f.write(initial_condition_content)
