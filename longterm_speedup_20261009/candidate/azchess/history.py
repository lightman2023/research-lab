"""Incremental input/history for standard chess; repetition keys are exact.

Uses python-chess's own _transposition_key (including legal en-passant and
castling rights). Historical frames are immutable while exploring children.
"""
from collections import Counter
import chess
import numpy as np
from .game import BOARD_PLANES, HISTORY_LENGTH, game_outcome

def piece_planes(board):
    planes=np.zeros((8,8,12),dtype=np.float32)
    for owner,color in enumerate((chess.WHITE,chess.BLACK)):
        for piece in range(1,7):
            for square in chess.scan_forward(board.pieces_mask(piece,color)):
                planes[square>>3,square&7,owner*6+piece-1]=1
    return planes

class HistoryEncoder:
    def __init__(self,board):
        if type(board) is not chess.Board:
            raise TypeError('Incremental history supports standard chess.Board')
        self.board=board
        walker=board.copy(stack=True);reverse_keys=[];recent=[]
        while True:
            reverse_keys.append(walker._transposition_key())
            if len(recent)<HISTORY_LENGTH:recent.append(piece_planes(walker))
            if not walker.move_stack:break
            walker.pop()
        self.keys=list(reversed(reverse_keys));self.counts=Counter();frames=[]
        for index,key in enumerate(self.keys):
            self.counts[key]+=1
            reverse_index=len(self.keys)-1-index
            if reverse_index<HISTORY_LENGTH:
                frames.append((recent[reverse_index],self.counts[key]))
        self.frames=frames;self.root_frames=len(frames)

    def push(self):
        key=self.board._transposition_key();self.keys.append(key);self.counts[key]+=1
        self.frames.append((piece_planes(self.board),self.counts[key]))

    def pop(self):
        key=self.keys.pop();self.counts[key]-=1
        if not self.counts[key]:del self.counts[key]
        self.frames.pop()

    def encode(self):
        board=self.board;player=board.turn
        state=np.zeros((8,8,BOARD_PLANES),dtype=np.float32)
        for i,(pieces,count) in enumerate(reversed(self.frames[-HISTORY_LENGTH:])):
            base=i*14
            if player==chess.WHITE:
                state[:,:,base:base+12]=pieces
            else:
                state[:,:,base:base+6]=pieces[::-1,:,6:12]
                state[:,:,base+6:base+12]=pieces[::-1,:,:6]
            state[:,:,base+12]=float(count>=2);state[:,:,base+13]=float(count>=3)
        state[:,:,112]=float(player==chess.WHITE)
        state[:,:,113]=min(len(board.move_stack),512)/512.
        state[:,:,114]=float(board.has_kingside_castling_rights(player))
        state[:,:,115]=float(board.has_queenside_castling_rights(player))
        state[:,:,116]=float(board.has_kingside_castling_rights(not player))
        state[:,:,117]=float(board.has_queenside_castling_rights(not player))
        state[:,:,118]=min(board.halfmove_clock,100)/100.
        return state

    def outcome(self):
        board=self.board
        if board.is_checkmate():return chess.Outcome(chess.Termination.CHECKMATE,not board.turn)
        if board.is_insufficient_material():return chess.Outcome(chess.Termination.INSUFFICIENT_MATERIAL,None)
        if not any(board.generate_legal_moves()):return chess.Outcome(chess.Termination.STALEMATE,None)
        if board.is_seventyfive_moves():return chess.Outcome(chess.Termination.SEVENTYFIVE_MOVES,None)
        count=self.counts[self.keys[-1]]
        if count>=5:return chess.Outcome(chess.Termination.FIVEFOLD_REPETITION,None)
        if count>=3:return chess.Outcome(chess.Termination.THREEFOLD_REPETITION,None)
        if board.is_fifty_moves():return chess.Outcome(chess.Termination.FIFTY_MOVES,None)
        return None

    def terminal_value(self):
        outcome=self.outcome()
        if outcome is None:return None
        if outcome.winner is None:return 0.
        return 1. if outcome.winner==self.board.turn else -1.
