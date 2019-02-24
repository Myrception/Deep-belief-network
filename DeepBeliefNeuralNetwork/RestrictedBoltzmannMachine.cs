using DeepBeliefNeuralNetwork.RBMComponents;
using System;
using System.Collections.Generic;

namespace DeepBeliefNeuralNetwork
{
    internal class RestrictedBoltzmannMachine
    {
        public List<RBMNeurons> VisualNeurons = new List<RBMNeurons>();
        public List<RBMNeurons> HiddenNeurons = new List<RBMNeurons>();
        public RBMBiasNeuron BiasNeuron = new RBMBiasNeuron();
        public double[] InitialLearningVector;
        public RBMWeightMatrix WeightMatrix { get; set; }
        private double _learningRate;
        public readonly int NumberOfVisualNeurons, NumberOfHiddenNeurons;

        //private List<RBMNeurons> _initialBinaryHiddenList;
        private double[] _initialHiddenStates;

        /// <summary>
        /// Konstruktor der Restricted Boltzmann Machines
        /// </summary>
        /// <param name="numberOfVisualNeurons">Anzahl er Sichtbaren Neuronen</param>
        /// <param name="numberOfHiddenNeurons">Anzahl der NIcht Sichtbaren Neuronen</param>
        /// <param name="learningRate">Die Lernrate</param>
        /// <param name="RND">Random Variable</param>
        internal RestrictedBoltzmannMachine(int numberOfVisualNeurons, int numberOfHiddenNeurons, double learningRate, ThreadSafeRandom RND)
        {
            int index = 0;
            NumberOfHiddenNeurons = numberOfHiddenNeurons;
            NumberOfVisualNeurons = numberOfVisualNeurons;
            WeightMatrix = new RBMWeightMatrix(numberOfVisualNeurons + numberOfHiddenNeurons, numberOfVisualNeurons + numberOfHiddenNeurons);
            for (int i = 0; i < numberOfVisualNeurons; i++)
            {
                VisualNeurons.Add(new RBMNeurons() { Index = index });
                index++;
            }
            for (int i = 0; i < numberOfHiddenNeurons; i++)
            {
                HiddenNeurons.Add(new RBMNeurons() { Index = index });
                index++;
            }
            _learningRate = learningRate;
            WeightMatrixRandom(numberOfVisualNeurons, numberOfHiddenNeurons, RND);
        }

        /// <summary>
        /// Hier findet das Greedy Layer Wise Training statt.
        /// </summary>
        /// <param name="trainingVector">Der zu trainierende Vektor</param>
        /// <param name="contrastiveDivergenceN">Anzahl an Iteratonen von CD</param>
        /// <param name="rnd">Random Variable</param>
        /// <returns></returns>
        internal double[] GreedyLearning(double[] trainingVector, int contrastiveDivergenceN, ThreadSafeRandom rnd)
        {
            InitialLearningVector = new double[trainingVector.Length];
            InitialLearningVector = trainingVector;
            RBMContrastiveDivergence training = new RBMContrastiveDivergence();
            double absError, squaredError;
            double[] OutputVector = new double[NumberOfHiddenNeurons];
            _initialHiddenStates = new double[HiddenNeurons.Count];
            for (int i = 0; i < trainingVector.Length; i++)
            {
                VisualNeurons[i].BinaryState = trainingVector[i];
            }
            absError = 0;
            squaredError = 0;
            foreach (var hiddenNeuron in HiddenNeurons) // Propagate Up
            {
                hiddenNeuron.BinaryState = training.CalculateHiddenBinaryState(BiasNeuron.BiasHidden, hiddenNeuron, WeightMatrix, VisualNeurons, NumberOfVisualNeurons, rnd);
            }
            for (int i = 0; i < HiddenNeurons.Count; i++)
            {
                _initialHiddenStates[i] = HiddenNeurons[i].Probability;
            }
            for (int i = 0; i < contrastiveDivergenceN; i++)
            {
                if (i == 0)
                {
                    foreach (var visualNeuron in VisualNeurons) // Propagate Down, Sample Visible (negative), for the data driven states
                    {
                        visualNeuron.BinaryState = training.CalculateReconstructionVisualBinaryStateDrivenByData(BiasNeuron.BiasVisible, visualNeuron, WeightMatrix, HiddenNeurons, NumberOfVisualNeurons, NumberOfHiddenNeurons, rnd);
                    }
                }
                else
                {
                    foreach (var visualNeuron in VisualNeurons) // Propagate Down, Sample Visible (negative), for the reconstruction driven states
                    {
                        visualNeuron.BinaryState = training.CalculateReconstructionVisualBinaryStateDrivenByReconstruction(BiasNeuron.BiasVisible, visualNeuron, WeightMatrix, HiddenNeurons, NumberOfVisualNeurons, NumberOfHiddenNeurons, rnd);
                    }
                }
                if (i == contrastiveDivergenceN - 1)
                {
                    foreach (var hiddenNeuron in HiddenNeurons) // Propagate Up, Sample Hidden (negative)
                    {
                        hiddenNeuron.BinaryState = training.CalculateHiddenBinaryStateWithProbability(BiasNeuron.BiasHidden, hiddenNeuron, WeightMatrix, VisualNeurons, NumberOfVisualNeurons, rnd);
                    }
                }
                else
                {
                    foreach (var hiddenNeuron in HiddenNeurons) // Propagate Up
                    {
                        hiddenNeuron.BinaryState = training.CalculateHiddenBinaryState(BiasNeuron.BiasHidden, hiddenNeuron, WeightMatrix, VisualNeurons, NumberOfVisualNeurons, rnd);
                    }
                }
            }
            training.CalculateWeightMatrixAlteration(_learningRate, WeightMatrix, VisualNeurons, HiddenNeurons, BiasNeuron,
                InitialLearningVector, _initialHiddenStates, NumberOfVisualNeurons, NumberOfHiddenNeurons + NumberOfVisualNeurons);
            for (int i = 0; i < trainingVector.Length; i++)
            {
                absError += Math.Abs(trainingVector[i] - VisualNeurons[i].BinaryState);//Absolute Fehler
                squaredError += Math.Pow(trainingVector[i] - VisualNeurons[i].BinaryState, 2);
            }
            Console.WriteLine("ABSError:" + absError + " " + "squaredError:" + squaredError);

            for (int i = 0; i < HiddenNeurons.Count; i++)
            {
                OutputVector[i] = HiddenNeurons[i].Probability;
            }
            return OutputVector;
        }

        //public double[] CalculateOutput(double[] inputvector,RestrictedBoltzmannMachine RBM, Random rnd)
        //{
        //    double[] outputvector = new double[RBM.HiddenNeurons.Count];
        //    RBMContrastiveDivergence training = new RBMContrastiveDivergence();
        //    for (int i = 0; i < RBM.VisualNeurons.Count; i++)
        //    {
        //        RBM.VisualNeurons[i].BinaryState = inputvector[i];
        //    }
        //    foreach (var hiddenNeuron in RBM.HiddenNeurons) // Propagate Up
        //    {
        //       hiddenNeuron.BinaryState = training.CalculateHiddenBinaryState(RBM.BiasNeuron.BiasHidden, hiddenNeuron, RBM.WeightMatrix, RBM.VisualNeurons, RBM.NumberOfVisualNeurons,rnd);
        //    }
        //    for (int i = 0; i < RBM.HiddenNeurons.Count; i++)
        //    {
        //        outputvector[i] = RBM.HiddenNeurons[i].Probability;
        //    }
        //    return outputvector;
        //}
        ///
        private void WeightMatrixRandom(int numberOfVisualNeurons, int numberOfHiddenNeurons, ThreadSafeRandom RND)
        {
            for (int i = numberOfVisualNeurons; i < numberOfVisualNeurons + numberOfHiddenNeurons; i++)
            {
                for (int j = 0; j < numberOfVisualNeurons; j++)
                {
                    WeightMatrix[j, i] = 2 * (RND.NextDouble() - 0.5);
                    WeightMatrix[i, j] = WeightMatrix[j, i];
                }
            }
        }
    }
}