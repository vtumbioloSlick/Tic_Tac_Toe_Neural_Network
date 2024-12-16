import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Scanner;

public class NeuralNetwork {

   public static NeuralNode[] inputNodes;
   public static NeuralNode[][] hiddenNodes;
   public static NeuralNode[] outputNodes;

   public static int inputSize;
   public static int hiddenSize;
   public static int hiddenLayers;
   public static int outputSize;


   public NeuralNetwork(int inputSize, int hiddenSize,
         int outputSize, int hiddenLayers) {

      this.inputSize = inputSize;
      this.hiddenSize = hiddenSize;
      this.hiddenLayers = hiddenLayers;
      this.outputSize = outputSize;
      setupNodes();
      setupNetworkConnections();

   }


   // INITIALIZE NODES
   public static void setupNodes() {

      // initalize layers
      inputNodes = new NeuralNode[inputSize];
      hiddenNodes = new NeuralNode[hiddenSize][hiddenLayers];
      outputNodes = new NeuralNode[outputSize];

      // input nodes
      for (int i = 0; i < inputSize; i++) {
         inputNodes[i] = new NeuralNode("Input" + String.valueOf(i), hiddenSize);
      }

      // hidden Nodes
      for (int i = 0; i < hiddenLayers - 1; i++) {
         for (int j = 0; j < hiddenSize; j++) {
            hiddenNodes[j][i] = new NeuralNode("hidden " + String.valueOf(i) + "_" + String.valueOf(j), hiddenSize);
         }
      }

      // last hidden Node
      int lastLayer = hiddenLayers - 1;
      for (int i = 0; i < hiddenSize; i++) {
         hiddenNodes[i][lastLayer] = new NeuralNode("Last Hidden", outputSize);
      }

      // output nodes
      for (int i = 0; i < outputSize; i++) {
         outputNodes[i] = new NeuralNode("OutputNode", 0);
      }

   }

   // CONNECT NODES
   public static void setupNetworkConnections() {

      // connects input layer to the first hidden layer
      for (int i = 0; i < inputNodes.length; i++) {

         for (int j = 0; j < hiddenSize; j++) {

            inputNodes[i].connectedNodes[j] = hiddenNodes[j][0];
         }
      }

      // connects the hidden layers
      for (int i = 0; i < hiddenLayers - 1; i++) {
         for (int j = 0; j < hiddenSize; j++) {

            for (int k = 0; k < hiddenSize; k++) {
               hiddenNodes[j][i].connectedNodes[k] = hiddenNodes[k][i + 1];
            }
         }
      }

      // connect last hidden layer to output layer

      int lastLayer = hiddenLayers - 1;
      for (int i = 0; i < hiddenSize; i++) {
         for (int j = 0; j < outputSize; j++) {
            hiddenNodes[i][lastLayer].connectedNodes[j] = outputNodes[j];
         }
      }

   }

   // TRAIN THE NETWORK
   public void train() {
      ArrayList<TicTacToeTrainingExample> trainingExamples =
            processExamples("New_Tic_Tac_Toe-Examples.txt");

      int epocs = 500;
      for (int i = 0; i < epocs; i++) {

         for (TicTacToeTrainingExample example : trainingExamples) {
            feedForward(example.question);
            backPropogation(example.answer);

         }

      }


   }

   public static void backPropogation(double[] answer) {
      updateOutputError(answer);
      updateHiddenError();
      updateWeights();
   }

   public static void updateWeights() {
      // learning rate
      double eta = 0.00111;

      // input later weights to first hidden layer
      for (NeuralNode node : inputNodes) {

         for (int i = 0; i < node.weights.length; i++) {
            NeuralNode hiddenNode = hiddenNodes[i][0];
            node.weights[i] += eta * hiddenNode.error * node.value;
         }
      }

      // hidden layer weights
      for (int i = 0; i < hiddenLayers - 1; i++) {
         for (int j = 0; j < hiddenSize; j++) {
            NeuralNode currentNode = hiddenNodes[j][i];
            for (int k = 0; k < hiddenSize; k++) {
               currentNode.weights[k] += eta * hiddenNodes[k][i + 1].error * currentNode.value;
            }
            currentNode.bias += eta * currentNode.error;
         }
      }

      // last hidden layer weights
      int lastLayer = hiddenLayers - 1;
      for (int i = 0; i < hiddenSize; i++) {
         NeuralNode currentNode = hiddenNodes[i][lastLayer];
         for (int j = 0; j < outputSize; j++) {
            currentNode.weights[j] += eta * outputNodes[j].error * currentNode.value;

         }
         currentNode.bias += eta * currentNode.error;
      }

      for (int i = 0; i < outputNodes.length; i++) {
         outputNodes[i].bias += eta * outputNodes[i].error;
      }

   }

   public static void updateHiddenError() {

      // delta h = dGelu * Sum (Wkh * delta-k)

      // Last hidden layer first
      int lastLayer = hiddenLayers  - 1;
      for (int i = 0; i < hiddenSize; i++) {
         NeuralNode currentNode = hiddenNodes[i][lastLayer];
//         double output = currentNode.value;
         double linearCombo = currentNode.bias;
         for (int k = 0; k <hiddenSize; k++) {
            linearCombo += hiddenNodes[k][lastLayer - 1].value * hiddenNodes[k][lastLayer - 1].weights[i];
         }

         double sum = 0;
         for (int j = 0; j < currentNode.weights.length; j++) {
            sum += currentNode.weights[j] * outputNodes[j].error;
         }
         double derivative = currentNode.derivative(linearCombo, 1);
         double error = derivative * sum;
         currentNode.setError(error);
      }

      // rest of the hidden layer
      for (int i = hiddenLayers - 2; i > 0; i--) {
         for (int j = 0; j < hiddenSize; j++) {
            NeuralNode currentNode = hiddenNodes[j][i];
            double linearCombo = currentNode.bias;
            for (int k = 0; k <hiddenSize; k++) {
               linearCombo += hiddenNodes[k][i - 1].value * hiddenNodes[k][i - 1].weights[j];
            }

            double sum = 0;
            for (int k = 0; k < currentNode.weights.length; k++) {
               sum += currentNode.weights[k] * hiddenNodes[k][i + 1].error;
            }
            double derivative = currentNode.derivative(linearCombo, 1);
            double error = derivative * sum;
            currentNode.setError(error);
         }
      }

      // first hidden layer
      for (int i = 0; i < hiddenSize; i++ ) {
         NeuralNode currentNode = hiddenNodes[i][0];
         double linearCombo = currentNode.bias;
         for (int j = 0; j < inputSize; j++) {
            linearCombo += inputNodes[j].value * inputNodes[j].weights[i];
         }

         double sum = 0;
         for (int k = 0; k < currentNode.weights.length; k++) {
            sum += currentNode.weights[k] * hiddenNodes[k][1].error;
         }
         double derivative = currentNode.derivative(linearCombo, 1);
         double error = derivative * sum;
         currentNode.setError(error);
      }

   }

   public static void updateOutputError(double[] answer) {

      // delta k = dGelu (Output) * (Tk - Ok)
      for (int i = 0; i < outputNodes.length; i++) {
         double output = outputNodes[i].value;

         double linearCombo = outputNodes[i].bias;
         for (int j = 0; j < hiddenSize; j++) {
            linearCombo += hiddenNodes[j][hiddenLayers - 1].value + hiddenNodes[j][hiddenLayers - 1].weights[i];
         }

         double target = answer[i];
         double derivative = outputNodes[i].derivative(linearCombo, 1);
         double error = derivative * (target - output);
         outputNodes[i].setError(error);
      }
   }
   // FEEDS THE EXAMPLE THROUGH THE NETWORK FROM INPUT TO OUTPUT

   public static void feedForward(double[] example) {

      // feed into input layer
      for (int i = 0; i < inputNodes.length; i++) {
         inputNodes[i].value = example[i];
      }

      // feed input into hidden layer
      for (int i = 0; i < hiddenSize; i++) {
         double linearCombo = 0.0;
         for (int j = 0; j < inputSize; j++) {
            double value = inputNodes[j].value * inputNodes[j].weights[i];
            linearCombo += value;
         }
         linearCombo += hiddenNodes[i][0].bias;
          hiddenNodes[i][0].value = hiddenNodes[i][0].gelu(linearCombo, 1); // 1 = Normal GELU
      }

      // feed through the hidden layer
      for (int i = 1; i < hiddenLayers; i++) {
         for (int j = 0; j < hiddenSize; j++) {
            double linearCombo = 0.0;
            for (int k = 0; k < hiddenSize; k++) {
               double value = hiddenNodes[k][i - 1].value * hiddenNodes[k][i - 1].weights[j];
               linearCombo += value;
            }
            linearCombo += hiddenNodes[j][i].bias;
            hiddenNodes[j][i].value = hiddenNodes[j][i].gelu(linearCombo, 1);

         }
      }

      // feed into the output layer
      int lastLayer = hiddenLayers -1;
      for (int i = 0; i < outputSize; i++) {
         double linearCombo = 0;
         for (int j = 0; j < hiddenSize; j++) {
            double value = hiddenNodes[j][lastLayer].value * hiddenNodes[j][lastLayer].weights[i];
            linearCombo += value;
         }
         linearCombo += outputNodes[i].bias;
         outputNodes[i].value = outputNodes[i].tanh(linearCombo);

      }


   }

   // PRINTS A SINGLE POSITION
   public static void printPosition(double[] position) {

      for (int i = 0; i < position.length; i++) {
         System.out.print(position[i] + " ");
      }
      System.out.println();

   }

   // PROCESSES EXAMPLES FROM THE SPECIFIED FILE
   // STRING IS CONVERTED TO Q AND A FORMAT
   public static ArrayList<TicTacToeTrainingExample> processExamples(String fileName){
      ArrayList<TicTacToeTrainingExample> trainingExamlpes = new ArrayList<>();

      try (Scanner scan = new Scanner(Paths.get(fileName))) {

         while (scan.hasNextLine()) {
            String line = scan.nextLine();
            TicTacToeTrainingExample example = processPosition(line);
            trainingExamlpes.add(example);
         }

      } catch (Exception e) {
         System.out.println(e);
      }
      return trainingExamlpes;
   }

   // CONVERTS A POSITION FROM STRING TO DOUBLE[] REPRESENTATION
   public static TicTacToeTrainingExample processPosition(String representation) {

      String[] parts = representation.split("\\| ");
      String[] question = parts[0].split(" ");
      String[] answer = parts[1].split(" ");
      double[] q = new double[question.length];
      double[] a = new double[answer.length];

      // process representation
      for (int i = 0; i < question.length; i++) {
         String q_s = question[i];
         String a_s = answer[i];
         // process question
         if (q_s.equals("_")) {
            q[i] = 0.0;
         } else if (q_s.equals("X")) {
            q[i] = -1.0;
         } else if (q_s.equals("O")) {
            q[i] = 1.0;
         } else {
            throw new ArithmeticException();
         }


      }
      // Variable input node way
      double[] ans = new double[inputSize];
      for (int i = 0; i < answer.length; i++) {
         ans[i] = Double.valueOf(answer[i]);
      }

      TicTacToeTrainingExample example = new TicTacToeTrainingExample(q,ans);
      return example;
   }

   public static void printBoard(char[][] board) {

      for (int i = 0; i < board[0].length; i++) {
         for (int j = 0; j < board[0].length; j++) {
            System.out.print( board[i][j] + " ");
         }
         System.out.println();
      }

   }

   public char[][] makePrediction(char[][] board) {

      double[] question = new double[board[0].length * board[0].length];
      int index = 0;
      for (int i = 0; i < board[0].length; i++) {
         for (int j = 0; j < board[0].length; j++) {
            if (board[i][j] == ' ') {
               question[index] = 0.0;
            } else if (board[i][j] == 'O') {
               question[index] = 1.0;
            } else if (board[i][j] == 'X') {
               question[index] = -1.0;
            } else {
               throw new ArithmeticException();
            }
            index++;
         }
      }
      feedForward(question);
      int[] legalMoves = legalPositions(question);

      double tracker = Double.NEGATIVE_INFINITY;
      int nodeTracker = -1;

      if (legalMoves.length == 0) {
         System.out.println("Zero");
      }

      for (int move : legalMoves) {
         if (Double.isNaN(outputNodes[move].value)) {
            System.out.println("NaN detected in outputNodes[" + move + "].value");
         }
      }


      for (int i = 0; i < legalMoves.length; i++) {

         if (outputNodes[legalMoves[i]].value > tracker) {
            tracker = outputNodes[legalMoves[i]].value;
            nodeTracker = i;
         }
      }

      double[] answer = new double[question.length];

      for (int i = 0; i < question.length; i++) {
         answer[i] = question[i];
      }

      // update board with new position, currently 1.0 = 'O'
      answer[legalMoves[nodeTracker]] = 1.0;

      char[][] predication = new char[board[0].length][board[0].length];

      int dubIndex = 0;

      for (int i = 0; i < predication[0].length; i++) {
         for (int j = 0; j < predication[0].length; j++) {

            if (answer[dubIndex] == 1.0) {
               predication[i][j] = 'O';
            } else if (answer[dubIndex] == 0.0) {
               predication[i][j] = ' ';
            } else if (answer[dubIndex] == -1.0) {
               predication[i][j] = 'X';
            } else {
               throw new ArithmeticException();
            }
            dubIndex++;
         }
      }
      return predication;
   }

   public static int[] legalPositions(double[] question) {

      int count = 0;
      for (int i = 0; i < question.length; i++) {
         if (question[i] == 0.0) {
            count++;
         }
      }
      int[] legalMoves = new int[count];
      int index = 0;
      for (int i = 0; i < question.length; i++) {
         if (question[i] == 0.0) {
            legalMoves[index] = i;
            index++;
         }
      }

      return legalMoves;
   }

}
