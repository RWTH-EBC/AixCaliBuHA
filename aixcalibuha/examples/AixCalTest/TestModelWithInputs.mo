within AixCalTest;
model TestModelWithInputs
  "Basic model for testing of calibration and sensitivity analysis"

    extends Modelica.Icons.Example;
   replaceable package Medium =
      Modelica.Media.Water.StandardWater
     constrainedby Modelica.Media.Interfaces.PartialMedium;
  Modelica.Fluid.Sources.MassFlowSource_T source_1(nPorts=1,
    redeclare package Medium = Medium,
    use_T_in=false,
    m_flow=0.5,
    use_m_flow_in=true,
    final T=313.15)
    annotation (Placement(transformation(extent={{-58,30},{-38,50}})));
  Modelica.Fluid.Sources.MassFlowSource_T source_2(
    nPorts=1,
    redeclare final package Medium = Medium,
    m_flow=m_flow_2,
    use_m_flow_in=true,
    final T=293.15)
              annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=180,
        origin={46,-38})));
  Modelica.Fluid.Sources.FixedBoundary sink_1(nPorts=1, redeclare package
      Medium = Medium,
    final p=Medium.p_default,
    final T=Medium.T_default,
    each final X=Medium.X_default)                      annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=180,
        origin={74,40})));
  Modelica.Fluid.Sources.FixedBoundary sink_2(nPorts=1, redeclare final package
      Medium = Medium,
    final p=Medium.p_default,
    final T=Medium.T_default,
    each final X=Medium.X_default)                      annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={-56,-38})));
  Modelica.Fluid.Pipes.DynamicPipe
                    heater(
    redeclare final package Medium = Medium,
    final use_T_start=true,
    redeclare model HeatTransfer =
        Modelica.Fluid.Pipes.BaseClasses.HeatTransfer.IdealFlowHeatTransfer,
    final nNodes=1,
    final use_HeatTransfer=true,
    final modelStructure=Modelica.Fluid.Types.ModelStructure.a_v_b,
    final length=2,
    final diameter=0.01,
    redeclare final model FlowModel =
        Modelica.Fluid.Pipes.BaseClasses.FlowModels.DetailedPipeFlow,
    final nParallel=1,
    final isCircular=true,
    final roughness=2.5e-5,
    final height_ab=0,
    final allowFlowReversal=heater.system.allowFlowReversal,
    each final X_start=Medium.X_default,
    final p_a_start=130000,
    final T_start=313.15)
    annotation (Placement(transformation(extent={{10,-10},{-10,10}},
        rotation=180,
        origin={0,40})));
  Modelica.Fluid.Pipes.DynamicPipe
                    heater1(
    redeclare final package Medium = Medium,
    final use_T_start=true,
    redeclare model HeatTransfer =
        Modelica.Fluid.Pipes.BaseClasses.HeatTransfer.IdealFlowHeatTransfer,
    final nNodes=1,
    final use_HeatTransfer=true,
    final modelStructure=Modelica.Fluid.Types.ModelStructure.a_v_b,
    final length=2,
    final diameter=0.01,
    redeclare final model FlowModel =
        Modelica.Fluid.Pipes.BaseClasses.FlowModels.DetailedPipeFlow,
    final nParallel=1,
    final isCircular=true,
    final roughness=2.5e-5,
    final height_ab=0,
    final allowFlowReversal=heater.system.allowFlowReversal,
    each final X_start=Medium.X_default,
    final T_start=Modelica.SIunits.Conversions.from_degC(20),
    final p_a_start=130000)
    annotation (Placement(transformation(extent={{10,-10},{-10,10}},
        rotation=0,
        origin={0,-38})));
  BaseClasses.HeatExchanger heatExchanger(final C=C)
    annotation (Placement(transformation(extent={{-10,-8},{10,12}})));
  Modelica.Blocks.Sources.Constant Gc_a(final k=heatConv_a)
    annotation (Placement(transformation(extent={{-58,2},{-42,18}})));
  Modelica.Blocks.Sources.Constant Gc_b(final k=heatConv_b) annotation (
      Placement(transformation(
        extent={{-8,-8},{8,8}},
        rotation=180,
        origin={54,-4})));
  parameter Real heatConv_b=500
                            "Constant output value" annotation (Evaluate=false);
  parameter Real heatConv_a=300
                            "Constant output value" annotation (Evaluate=false);
  parameter Modelica.SIunits.HeatCapacity C=8000
                                            "Heat capacity of element (= cp*m)"
    annotation (Evaluate=false);
  inner Modelica.Fluid.System system(
    final g=Modelica.Constants.g_n,
    use_eps_Re=false,
    final m_flow_small=1e-2,
    final p_ambient=101325,
    final T_ambient=293.15,
    final dp_small=1)
    annotation (Placement(transformation(extent={{-100,80},{-80,100}})));
  parameter Modelica.Media.Interfaces.PartialMedium.MassFlowRate m_flow_2=0.03
    "Fixed mass flow rate going out of the fluid port" annotation (Evaluate=false);
  Modelica.Blocks.Sources.Constant m_flow_sink(final k=m_flow_2) annotation (
      Placement(transformation(
        extent={{-8,-8},{8,8}},
        rotation=180,
        origin={114,-28})));
  Modelica.Blocks.Logical.Switch switch1 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={-84,48})));
  Modelica.Blocks.Logical.Switch switch2 annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=180,
        origin={78,-46})));
  Modelica.Blocks.Sources.Constant const_zero(final k=0)
    annotation (Placement(transformation(extent={{-148,16},{-132,32}})));
  Modelica.Blocks.Sources.Constant m_flow_source(final k=0.5)
    annotation (Placement(transformation(extent={{-124,56},{-108,72}})));
  Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor temperatureSensor
    annotation (Placement(transformation(extent={{16,16},{30,30}})));
  Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor temperatureSensor1
    annotation (Placement(transformation(extent={{14,-32},{28,-22}})));
  Modelica.Blocks.Interfaces.RealInput device_on
    annotation (Placement(transformation(extent={{-138,-62},{-98,-22}})));
  Modelica.Blocks.Logical.GreaterThreshold greaterThreshold
    annotation (Placement(transformation(extent={{-80,-92},{-60,-72}})));
equation
  connect(source_1.ports[1], heater.port_a)
    annotation (Line(points={{-38,40},{-10,40}}, color={0,127,255}));
  connect(heater.port_b, sink_1.ports[1])
    annotation (Line(points={{10,40},{64,40}}, color={0,127,255}));
  connect(source_2.ports[1], heater1.port_a) annotation (Line(points={{36,-38},
          {10,-38}},                  color={0,127,255}));
  connect(heater1.port_b, sink_2.ports[1])
    annotation (Line(points={{-10,-38},{-46,-38}}, color={0,127,255}));
  connect(heater.heatPorts[1], heatExchanger.port_a) annotation (Line(points={{0.1,
          35.6},{0.1,23.8},{0,23.8},{0,12}}, color={127,0,0}));
  connect(heater1.heatPorts[1], heatExchanger.port_b) annotation (Line(points={{
          -0.1,-33.6},{-0.1,-20.8},{0,-20.8},{0,-8}}, color={127,0,0}));
  connect(Gc_a.y, heatExchanger.Gc_a) annotation (Line(points={{-41.2,10},{-28,10},
          {-28,8.8},{-12,8.8}}, color={0,0,127}));
  connect(Gc_b.y, heatExchanger.Gc_b) annotation (Line(points={{45.2,-4},{30,-4},
          {30,-5},{12,-5}}, color={0,0,127}));
  connect(switch1.y, source_1.m_flow_in)
    annotation (Line(points={{-73,48},{-58,48}}, color={0,0,127}));
  connect(switch2.y, source_2.m_flow_in)
    annotation (Line(points={{67,-46},{56,-46}}, color={0,0,127}));
  connect(m_flow_sink.y, switch2.u1)
    annotation (Line(points={{105.2,-28},{102,-28},{102,-38},{90,-38}},
                                                     color={0,0,127}));
  connect(m_flow_source.y, switch1.u1) annotation (Line(points={{-107.2,64},{
          -104,64},{-104,56},{-96,56}},  color={0,0,127}));
  connect(const_zero.y, switch1.u3) annotation (Line(points={{-131.2,24},{-122,
          24},{-122,40},{-96,40}},   color={0,0,127}));
  connect(const_zero.y, switch2.u3) annotation (Line(points={{-131.2,24},{-120,
          24},{-120,-70},{120,-70},{120,-54},{90,-54}},   color={0,0,127}));
  connect(temperatureSensor.port, heater.heatPorts[1]) annotation (Line(points=
          {{16,23},{12,23},{12,22},{0.1,22},{0.1,35.6}}, color={191,0,0}));
  connect(temperatureSensor1.port, heater1.heatPorts[1]) annotation (Line(
        points={{14,-27},{12,-27},{12,-28},{-0.1,-28},{-0.1,-33.6}}, color={191,
          0,0}));
  connect(greaterThreshold.y, switch1.u2) annotation (Line(points={{-59,-82},{
          -50,-82},{-50,-62},{-108,-62},{-108,48},{-96,48}}, color={255,0,255}));
  connect(greaterThreshold.y, switch2.u2) annotation (Line(points={{-59,-82},{
          134,-82},{134,-46},{90,-46}}, color={255,0,255}));
  connect(device_on, greaterThreshold.u) annotation (Line(points={{-118,-42},{
          -96,-42},{-96,-82},{-82,-82}}, color={0,0,127}));
  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
        coordinateSystem(preserveAspectRatio=false)),
    experiment(StopTime=3600, Interval=1));
end TestModelWithInputs;
