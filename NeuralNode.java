import java.util.Random;
public class NeuralNode {

   // GLOBAL VARIABLES

   public double[] weights;
   final public static Random randy = new Random();
   public double bias = randy.nextDouble() - 0.5; // Range [-0.5, 0.5]
   public NeuralNode[] connectedNodes;
   String nodeName;
   double value;
   double error;

//   public ActivationFunction activationFunction;
   // constructor sets name, and connections
   public NeuralNode(final String nodeName, int connectedNodes) {
      this.nodeName = nodeName;
      this.connectedNodes = new NeuralNode[connectedNodes];
//      this.activationFunction = ActivationFactory.getActivationFunction(activationFunction);

      weights = new double[connectedNodes];
      initalizeWeights();
   }

   // error settor function
   public void setError(final double error) {this.error = error;}

   public void initalizeWeights() {
      for (int i = 0; i < weights.length; i++) {
         weights[i] = randy.nextDouble() - 0.5; // RANGE [-0.5 , 0.5]
      }
   }


   // GELU ACTIVATION FUNCTION AND HELPER METHODS

   // if s = 1, this is standard gelu function
   public double gelu(double x, double s) {

      double root2 = Math.sqrt(2);
      double coeff = x / 2;
      double value = coeff * (1 + erf(s * x / root2));
      //      System.out.println(value);
      return value;
   }

   // ERROR FUNCTION AND HELPER METHODS


   // fractional error less than x.xx * 10 ^ -4.
   // Algorithm 26.2.17 in Abromowitz and Stegun, Handbook of Mathematical.
   public static double erf(double z) {
      double t = 1.0 / (1.0 + 0.47047 * Math.abs(z));
      double poly = t * (0.3480242 + t * (-0.0958798 + t * (0.7478556)));
      double ans = 1.0 - poly * Math.exp(-z*z);
      if (z >= 0) return  ans;
      else        return -ans;
   }

   // cumulative normal distribution
   public static double Phi(double z) {
      return 0.5 * (1.0 + erf(z / (Math.sqrt(2.0))));
   }

   // Derivative of GELU
   public double derivative(double x, double s) {
      // splitting equation into parts for ease and readability
      double coeff = (s * x) / Math.sqrt(2 * Math.PI);
      double exp = Math.pow(Math.E, -Math.pow((s * x), 2) / 2);
      double p2 = 0.5 * (1 + erf((s * x) / Math.sqrt(2)));
      double value = coeff * exp + p2;
      return value;
   }


   // SOFTMAX

   public static double[] softmax(NeuralNode[] nodes) {
      // Step 1: Calculate the exponential for each node's value
      double[] expValues = new double[nodes.length];
      double sumExp = 0.0;
      for (int i = 0; i < nodes.length; i++) {
         expValues[i] = Math.exp(nodes[i].value);
         sumExp += expValues[i];
      }

      // Step 2: Normalize the values to get probabilities
      double[] softmaxValues = new double[nodes.length];
      for (int i = 0; i < nodes.length; i++) {
         softmaxValues[i] = expValues[i] / sumExp;
      }

      return softmaxValues;
   }

   public static double[][] softmaxDerivative(double[] softmaxValues) {
      int n = softmaxValues.length;
      double[][] jacobianMatrix = new double[n][n];

      // Compute the Jacobian matrix
      for (int i = 0; i < n; i++) {
         for (int j = 0; j < n; j++) {
            if (i == j) {
               // Diagonal elements
               jacobianMatrix[i][j] = softmaxValues[i] * (1 - softmaxValues[i]);
            } else {
               // Off-diagonal elements
               jacobianMatrix[i][j] = -softmaxValues[i] * softmaxValues[j];
            }
         }
      }

      return jacobianMatrix;
   }


   // Sigmoid activation function
   public static double sigmoid(double x) {
      return 1.0 / (1.0 + Math.exp(-x));
   }

   // Derivative of the sigmoid function
   public static double sigmoidDerivative(double sigmoidValue) {
      return sigmoidValue * (1 - sigmoidValue);
   }

   public double tanh(double x) {
      return Math.tanh(x);
   }

   // Compute the derivative of the tanh function
   public double tanhDerivative(double x) {
      double tanhValue = Math.tanh(x);

      if (1.0 - tanhValue * tanhValue == 0) {
         System.out.println("hello :     0");
      }
      return 1.0 - tanhValue * tanhValue;
   }

}
