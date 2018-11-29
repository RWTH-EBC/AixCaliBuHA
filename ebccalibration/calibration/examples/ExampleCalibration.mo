within ;
model ExampleCalibration
  "Example model to demonstrate the calibration process"
  parameter Real amplitude=1 "Amplitude of sine wave" annotation(Evaluate=false);
  parameter Modelica.SIunits.Frequency freqHz=1 "Frequency of sine wave" annotation(Evaluate=false);
  Modelica.Blocks.Sources.Sine sine(amplitude=amplitude, freqHz=freqHz)
    annotation (Placement(transformation(extent={{-88,6},{-62,32}})));
  Modelica.Blocks.Sources.Pulse pulse(
    amplitude=2,
    offset=-1,
    period=4)
    annotation (Placement(transformation(extent={{46,-26},{74,2}})));

  Modelica.Blocks.Sources.Trapezoid
                               trapezoid(
    amplitude=2,
    offset=-1,
    rising=1,
    width=2,
    falling=1,
    period=4)
    annotation (Placement(transformation(extent={{46,34},{74,62}})));
  annotation (uses(Modelica(version="3.2.2")), Diagram(graphics={Text(
          extent={{26,110},{96,54}},
          lineColor={28,108,200},
          textString="\"Measurements\""), Text(
          extent={{-100,76},{-36,24}},
          lineColor={28,108,200},
          textString="\"Simulation\"")}));
end ExampleCalibration;
