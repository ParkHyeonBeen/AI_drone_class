$fs = 0.5;
$fa = 1;

pillar_h = 70;

cylinder (d=20, h=0.3);

linear_extrude (height = pillar_h)
difference () {
    circle (d=10);
    circle (d=10 - 0.4 * 2);
}
