from net import act
import net

############################################################################################
# act parameters order
# fx: horizontal direction of movement.
# fy: vertical direction of movement.
# speed: velocity of movement, let it on 0.5 speed.
# crouch: let it crouch (agachar in portuguese).
# jump: let it jump (pular in portuguese).
# l (left): left rotation of the head.
# r (right): right rotation of the head.
# up: vertical to up rotation of the head.
# d (down): vertical to down rotation of the head.
# ps: push
# rf: release the flag.
# gf: get the flag.
# ws: apply walk speed. 
# rs: restart
##############################################################################################
def walk(speed=0.5):
	act(0.0, speed)

def run():
	walk(3.0)
	
def walk_in_circle(speed=0.5):
	act(speed, speed)

def crouch():
	act(0.0, 0.0, 0.0, True, False, 0.0)

def jump():
	act(0.0, 0.0, 0.0, False, True, 0.0)
	
def see_around_by_left(speed=0.5):
	act(0.0, 0.0, 0.0, False, False, speed)
	
def see_around_by_right(speed=0.5):
	act(0.0, 0.0, 0.0, False, False, 0.0, speed)
	
def see_around_up(speed=0.5):
	act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, speed)
	
def see_around_down(speed=0.5):
	act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, speed)

def push():
	act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, 0.0, True)
	
def reset_state():
	act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, 0.0, False, True)

def get_pickup():
	act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, 0.0, False, False, True)

def restart():
	act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, 0.0, False, False, False, True, True)

def pause():
	act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, 0.0, False, False, False, True, False, True)

def resume():
	act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, 0.0, False, False, False, True, False, False, True)

	