# COMP30024_AI_GamePlayingAgent
An AI game-playing agent written in Python which plays a game of Cachex (an alternative to the board game Hex, including a capture mechanism)
<br>
<h4> Completed in May 2022 (Semester 2) </h4>
<hr>


<b>Context and Functionality:</b>
Cachex is a perfect-information two-player game played on an n × n rhombic, hexagonally tiled board, based on the strategy game Hex. Two players (named Red and Blue) compete, with the goal to form a connection between the opposing sides of the board corresponding to their respective color. The primary task was to design and implement a program to play the game of Cachex. That is, given information about the evolving state of the game, your program will decide on an action to take on each of its turns. The assessors have provided a driver program (titled 'referee') which communicates with the player agents to execute all administrative functions to run the program. The 'player' class implements the specific game-playing strategies. This assignment also includes a report which discusses the strategies we used to play the game and the algorithms implmented.

<b>Topics of focus:</b>
The aim of this project is to (1) practice applying game-playing techniques discussed in the lectures (in this case; depth-limited Minimax with alpha-beta pruning), (2) develop your own strategies for playing Cachex, and (3) conduct research into more advanced algorithmic game-playing techniques to extend the functionality of our agent.
<hr>

See 'Cachex_Rules' for a full explanation of the rules of the game; including the gameplay sequence, the capturing mechanism (unique to Cachex); and the conditions required to terminate a game.

See 'Cachex_Agent_Specification' to understand the specific functionality of the program, including the control flow (between the referee and player class), as well as the program constraints to aid in efficient automatic testing.

<b>To run this console application:</b>
<ol>
  <li> Download the folders 'Referee', 'auto-player' and 'user-input-player' (these modules include Python code files) and place these folders in a directory </li>
  <li> Navigate to the working directory to play a game using: <code>python -m referee "n" "red module" "blue module"</code> <br> 
    <em>DO NOT INCLUDE THE DOUBLE QUOTES ("") when typing out this command e.g. only use [ python n red-module blue-module" </em> <br> 
  <b>Where:</b> python is the game of a Python 3.6 interpreter, "<n>" is the size of the game board and "<red module>" and "<blue module>" are the names of modules contining the classes "Player" to be used for Red and Blue respectively.</li>
</ol>

<b>A few notes:</b>
<ul>
  <li> <code>Referee</code> is the referee class which plays the game </li>
  <li> <code>auto-player</code> is the player class which implements the algorithms and strategies we used to play the game automatically. Use <code> python n auto-player auto-player </code> to have a completely automatic game </li>
  <li> <code>user-input-player</code> is the player class to be used if you want to play the game manually. Subsitute <code> user-input-player </code> for either "Red" or "Blue" or both to have a manual game </li>
</ul>
