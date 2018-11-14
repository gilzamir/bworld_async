import socket
from threading import Thread
import time
import sys

ACT_PORT = 8881
PERCEPT_PORT = 8888
HOST = "127.0.0.1"
PERCEPT_BUFFER_SIZE = 4000

def create_udp():
	return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


UDP = create_udp()

def percept():
	return receive_data(HOST, PERCEPT_PORT, UDP)

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
# rf: reset state.
# gf: get the pickup.
# ws: apply walk speed. 
# rs: restart
##############################################################################################
def act(fx, fy, speed=0.5, crouch=False, jump=False, l=0.0, r=0.0, u=0.0, d=0.0, ps=False, rf=False, gf=False, ws=True, rs=False, pause=False, resume=False):
	send_command(HOST, ACT_PORT, UDP, fx, fy, speed, crouch, jump, l, r, u, d, ps, rf, gf, ws, rs, pause, resume)

##############################################
## Close Percept-Act cycle				   ##
##############################################
def close():
	UDP.close()

def open():
	open_receive(HOST, PERCEPT_PORT, UDP)

def open_receive(HOST, PORT, sock):
	server_address = (HOST, PORT)
	print('starting up on %s port %s'%server_address)
	sock.bind(server_address)
	
def receive_data(HOST, PORT, sock):
	# Bind the socket to the port
	return recvall(sock)

def recvall(sock):
	data=bytearray(PERCEPT_BUFFER_SIZE)
	try:
		data, addr=sock.recvfrom(PERCEPT_BUFFER_SIZE)
		return (data[0:50], data[50::])
	except:
		e = sys.exc_info()[1]
		print("Error: %s\n"%e)
		return None

############################################################################################
# Send commands to remote controller of the player.
# HOST: remote host
# PORT: communication port
# socket: socket object to communication
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
# rf: reset state.
# gf: get the pickup.
# ws: apply walk speed. 
##############################################################################################
def send_command(HOST, PORT, socket, fx, fy, speed, crouch, jump, l, r, u, d, ps, rf, gf, ws, rs=False, pause = False, resume = False):
	command = "%f;%f;%f;%r;%r;%f;%f;%f;%f;%r;%r;%r;%r;%r;%r;%r"%(fx, fy, speed, crouch, jump, l, r, u, d, ps, rf, gf, ws, rs, pause, resume)
	#print(command)
	dest = (HOST, PORT)
	#print('sending to %s port %s'%dest)
	socket.sendto (command.encode(), dest);
