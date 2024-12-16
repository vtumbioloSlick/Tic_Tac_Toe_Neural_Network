import java.util.Scanner;
public class NNDriver {

   public static void main (String[] args) {

      NeuralNetwork net = new NeuralNetwork(9, 50, 9, 4);
      net.train();

      while (true) {
         char[][] board = newBoard();
         while (true) {

            // AI Move
            board = net.makePrediction(board);
            printBoard(board);
            if (isGameOver(board)) {
               break;
            }

            playerMove(board, 'X');
            printBoard(board);
            if (isGameOver(board)) {
               break;
            }

         }

      }

   }


   public static void printBoard(char[][] board) {
      for (int i = 0; i < board[0].length; i++) {
         for (int j = 0; j < board[0].length; j++) {
            System.out.print(board[i][j] + " ");
         }
         System.out.println();
      }
   }

   public static void playerMove(char[][] board, char player) {

      Scanner sc = new Scanner(System.in);
      System.out.println("Enter a move p1,p2. Example \"2,2\" is bottom right");
      String[] parts = sc.nextLine().split(",");
      board[Integer.valueOf(parts[0])][Integer.valueOf(parts[1])] = player;
   }


   public static char[][] newBoard() {
      return new char[][] {
            {' ', ' ', ' '},
            {' ', ' ', ' '},
            {' ', ' ', ' '}
      };
   }

   public static boolean isGameOver(char[][] board) {
      // Check rows and columns for a winner
      for (int i = 0; i < 3; i++) {
         // Check row
         if (board[i][0] == board[i][1] && board[i][1] == board[i][2] && board[i][0] != ' ') {
            return true;
         }

         // Check column
         if (board[0][i] == board[1][i] && board[1][i] == board[2][i] && board[0][i] != ' ') {
            return true;
         }
      }

      // Check diagonals for a winner
      if (board[0][0] == board[1][1] && board[1][1] == board[2][2] && board[0][0] != ' ') {
         return true;
      }

      if (board[0][2] == board[1][1] && board[1][1] == board[2][0] && board[0][2] != ' ') {
         return true;
      }

      // Check if the board is full (draw)
      boolean isDraw = true;
      for (int i = 0; i < 3; i++) {
         for (int j = 0; j < 3; j++) {
            if (board[i][j] == ' ') {
               isDraw = false;
               break;
            }
         }
      }

      if (isDraw) {
         return true;
      }

      // Game is still ongoing
      return false;
   }

}
