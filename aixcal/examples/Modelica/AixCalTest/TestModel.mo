within AixCalTest;
model TestModel
  "Basic model for testing of calibration and sensitivity analysis"

    extends Modelica.Icons.Example;
   replaceable package Medium =
      Modelica.Media.Water.StandardWater
     constrainedby Modelica.Media.Interfaces.PartialMedium;
  Modelica.Fluid.Sources.MassFlowSource_T source_1(nPorts=1,
    redeclare package Medium = Medium,
    use_T_in=false,
    m_flow=0.5,
    T=313.15)
    annotation (Placement(transformation(extent={{-90,30},{-70,50}})));
  Modelica.Fluid.Sources.MassFlowSource_T source_2(
    nPorts=1,
    redeclare final package Medium = Medium,
    m_flow=m_flow_2,
    T=293.15) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=180,
        origin={74,-38})));
  Modelica.Fluid.Sources.FixedBoundary sink_1(nPorts=1, redeclare package
      Medium = Medium)                                  annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=180,
        origin={74,40})));
  Modelica.Fluid.Sources.FixedBoundary sink_2(nPorts=1, redeclare final package
      Medium = Medium)                                  annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={-78,-38})));
  Modelica.Fluid.Pipes.DynamicPipe
                    heater(
    redeclare package Medium = Medium,
    use_T_start=true,
    length=2,
    redeclare model HeatTransfer =
        Modelica.Fluid.Pipes.BaseClasses.HeatTransfer.IdealFlowHeatTransfer,
    diameter=0.01,
    nNodes=1,
    redeclare model FlowModel =
        Modelica.Fluid.Pipes.BaseClasses.FlowModels.DetailedPipeFlow,
    use_HeatTransfer=true,
    modelStructure=Modelica.Fluid.Types.ModelStructure.a_v_b,
    p_a_start=130000,
    T_start=313.15)
    annotation (Placement(transformation(extent={{10,-10},{-10,10}},
        rotation=180,
        origin={0,40})));
  Modelica.Fluid.Pipes.DynamicPipe
                    heater1(
    redeclare package Medium = Medium,
    use_T_start=true,
    redeclare model HeatTransfer =
        Modelica.Fluid.Pipes.BaseClasses.HeatTransfer.IdealFlowHeatTransfer,
    diameter=0.01,
    nNodes=1,
    redeclare model FlowModel =
        Modelica.Fluid.Pipes.BaseClasses.FlowModels.DetailedPipeFlow,
    use_HeatTransfer=true,
    modelStructure=Modelica.Fluid.Types.ModelStructure.a_v_b,
    final nParallel=1,
    final length=2,
    p_a_start=130000,
    T_start=Modelica.SIunits.Conversions.from_degC(20))
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
  parameter Real heatConv_b=233
                            "Constant output value" annotation (Evaluate=false);
  parameter Real heatConv_a=125
                            "Constant output value" annotation (Evaluate=false);
  parameter Modelica.SIunits.HeatCapacity C=5432
                                            "Heat capacity of element (= cp*m)"
    annotation (Evaluate=false);
  inner Modelica.Fluid.System system
    annotation (Placement(transformation(extent={{-100,80},{-80,100}})));
  parameter Modelica.Media.Interfaces.PartialMedium.MassFlowRate m_flow_2=0.02
    "Fixed mass flow rate going out of the fluid port" annotation (Evaluate=false);
equation
  connect(source_1.ports[1], heater.port_a)
    annotation (Line(points={{-70,40},{-10,40}}, color={0,127,255}));
  connect(heater.port_b, sink_1.ports[1])
    annotation (Line(points={{10,40},{64,40}}, color={0,127,255}));
  connect(source_2.ports[1], heater1.port_a) annotation (Line(points={{64,-38},
          {10,-38}},                  color={0,127,255}));
  connect(heater1.port_b, sink_2.ports[1])
    annotation (Line(points={{-10,-38},{-68,-38}}, color={0,127,255}));
  connect(heater.heatPorts[1], heatExchanger.port_a) annotation (Line(points={{0.1,
          35.6},{0.1,23.8},{0,23.8},{0,12}}, color={127,0,0}));
  connect(heater1.heatPorts[1], heatExchanger.port_b) annotation (Line(points={{
          -0.1,-33.6},{-0.1,-20.8},{0,-20.8},{0,-8}}, color={127,0,0}));
  connect(Gc_a.y, heatExchanger.Gc_a) annotation (Line(points={{-41.2,10},{-28,10},
          {-28,8.8},{-12,8.8}}, color={0,0,127}));
  connect(Gc_b.y, heatExchanger.Gc_b) annotation (Line(points={{45.2,-4},{30,-4},
          {30,-5},{12,-5}}, color={0,0,127}));
  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
        coordinateSystem(preserveAspectRatio=false)),
    experiment(StopTime=3600, Interval=1));
end TestModel;
