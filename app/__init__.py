import os, stripe, json
from datetime import datetime
from tokenize import String

import np as np
from flask import Flask, render_template, redirect, url_for, flash, request, abort
from flask_bootstrap import Bootstrap

from .admin.forms import AddItemForm
from .forms import LoginForm, RegisterForm, ReviewsForm
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, current_user, login_required, logout_user
from .db_models import db, User, Item, Review_item
from itsdangerous import URLSafeTimedSerializer
from .funcs import mail, send_confirmation_email, fulfill_order
from dotenv import load_dotenv
from .admin.routes import admin
from os.path import join, dirname, realpath
load_dotenv()
app = Flask(__name__)
app.register_blueprint(admin)

app.config["SECRET_KEY"] = '12345' #os.environ["SECRET_KEY"]
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///test.db" #os.environ["DB_URI"]
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_USERNAME'] ="alinababer@gmail.com" #os.environ["EMAIL"]
app.config['MAIL_PASSWORD'] = 'aleenababer046' #os.environ["PASSWORD"]
app.config['MAIL_SERVER'] = "smtp.googlemail.com"
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_PORT'] = 587

app.config['UPLOAD_FOLDER'] = 'static\\uploads'
stripe.api_key = os.getenv("STRIPE_PRIVATE") #os.environ["STRIPE_PRIVATE"]
Bootstrap(app)
db.init_app(app)
mail.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)

with app.app_context():
	db.create_all()

@app.context_processor
def inject_now():
	""" sends datetime to templates as 'now' """
	return {'now': datetime.utcnow()}

@login_manager.user_loader
def load_user(user_id):
	return User.query.get(user_id)

@app.route("/")
def index():
	items = Item.query.all()
	print(len(items))
	rate = []
	rating= [ ]
	reviews = []
	finalrating=[]
	reviewsdataframe= pd.DataFrame(columns=['rating'])
	resultrating = dict()
	ab=0
	for item in items:
		print(item)
		ab=ab+1
		key= item.name
		rate=[]
		print(len(Review_item.query.filter_by(itemid=int(item.id)).all()))
		if len(Review_item.query.filter_by(itemid=int(item.id)).all())>0:
			reviewslist = Review_item.query.filter_by(itemid=int(item.id)).all()
			for review in reviewslist:
				review = str(review)
				reviewlist = review.split(",")
				print(reviewlist)
				if reviewlist[-1] != None:
					reviewsdataframe.append([int(reviewlist[-1])])
					print(reviewsdataframe)
					rate.append([int(reviewlist[-1])])
			finalrating = int(np.mean(rate))
		elif len(Review_item.query.filter_by(itemid=int(item.id)).all()) == 0:
			finalrating =0
		raterange = []
		print(finalrating)
		if finalrating > 0:
			for i in range(0, finalrating):
				raterange.append(i)
			case = {key: raterange}
			resultrating.update(case)
		else:
			raterange = []
			case = {key: raterange}
			resultrating.update(case)
	print(resultrating)
	print(app.url_map)
	return render_template("index.html", items=items,rating=resultrating)

@app.route("/home")
def home():
	items = Item.query.all()
	print(len(items))
	rate = []
	rating= [ ]
	reviews = []
	finalrating=[]
	temprate=[]
	resultrating = dict()
	ab=0
	reviewsdataframe= pd.DataFrame(columns=['rating'])
	resultrating = dict()
	ab=0
	for item in items:
		print(item)
		ab=ab+1
		key= item.name
		rate=[]
		print(len(Review_item.query.filter_by(itemid=int(item.id)).all()))
		if len(Review_item.query.filter_by(itemid=int(item.id)).all())>0:
			reviewslist = Review_item.query.filter_by(itemid=int(item.id)).all()
			for review in reviewslist:
				review = str(review)
				reviewlist = review.split(",")
				print(reviewlist)
				if reviewlist[-1] != None:
					reviewsdataframe.append([int(reviewlist[-1])])
					print(reviewsdataframe)
					rate.append([int(reviewlist[-1])])
			finalrating = int(np.mean(rate))
		elif len(Review_item.query.filter_by(itemid=int(item.id)).all()) == 0:
			finalrating = 0
		raterange=[]
		print(finalrating)
		if finalrating > 0:
			for i in range(0, finalrating):
				raterange.append(i)
			case = {key: raterange}
			resultrating.update(case)
		else:
			raterange = []
			case = {key: raterange}
			resultrating.update(case)
	print(resultrating)
	return render_template("home.html", items=items,rating=resultrating)

@app.route("/login", methods=['POST', 'GET'])
def login():
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	form = LoginForm()
	if form.validate_on_submit():
		email = form.email.data
		user = User.query.filter_by(email=email).first()
		if user == None:
			flash(f'User with email {email} doesn\'t exist!<br> <a href={url_for("register")}>Register now!</a>', 'error')
			return redirect(url_for('login'))
		elif check_password_hash(user.password, form.password.data):
			login_user(user)
			return redirect(url_for('home'))
		else:
			flash("Email and password incorrect!!", "error")
			return redirect(url_for('login'))
	return render_template("login.html", form=form)

@app.route("/register", methods=['POST', 'GET'])
def register():
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	form = RegisterForm()
	if form.validate_on_submit():
		user = User.query.filter_by(email=form.email.data).first()
		if user:
			flash(f"User with email {user.email} already exists!!<br> <a href={url_for('login')}>Login now!</a>", "error")
			return redirect(url_for('register'))
		new_user = User(name=form.name.data,
						email=form.email.data,
						password=generate_password_hash(
									form.password.data,
									method='pbkdf2:sha256',
									salt_length=8),
						phone=form.phone.data)
		db.session.add(new_user)
		db.session.commit()
		# send_confirmation_email(new_user.email)
		flash('Thanks for registering! You may login now.', 'success')
		return redirect(url_for('login'))
	return render_template("register.html", form=form)

@app.route('/confirm/<token>')
def confirm_email(token):
	try:
		confirm_serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
		email = confirm_serializer.loads(token, salt='email-confirmation-salt', max_age=3600)
	except:
		flash('The confirmation link is invalid or has expired.', 'error')
		return redirect(url_for('login'))
	user = User.query.filter_by(email=email).first()
	if user.email_confirmed:
		flash(f'Account already confirmed. Please login.', 'success')
	else:
		user.email_confirmed = True
		db.session.add(user)
		db.session.commit()
		flash('Email address successfully confirmed!', 'success')
	return redirect(url_for('login'))

@app.route("/logout")
@login_required
def logout():
	logout_user()
	items = Item.query.all()
	items = Item.query.all()
	print(len(items))
	rate = []
	rating = []
	reviews = []
	finalrating = []
	reviewsdataframe = pd.DataFrame(columns=['rating'])
	resultrating = dict()
	ab = 0
	for item in items:
		print(item)
		ab = ab + 1
		key = item.name
		rate = []
		print(len(Review_item.query.filter_by(itemid=int(item.id)).all()))
		if len(Review_item.query.filter_by(itemid=int(item.id)).all()) > 0:
			reviewslist = Review_item.query.filter_by(itemid=int(item.id)).all()
			for review in reviewslist:
				review = str(review)
				reviewlist = review.split(",")
				print(reviewlist)
				if reviewlist[-1] != None:
					reviewsdataframe.append([int(reviewlist[-1])])
					print(reviewsdataframe)
					rate.append([int(reviewlist[-1])])
			finalrating = int(np.mean(rate))
		elif len(Review_item.query.filter_by(itemid=int(item.id)).all()) == 0:
			finalrating = 0
		raterange = []
		print(finalrating)
		if finalrating >0:
			for i in range(0, finalrating):
				raterange.append(i)
			case = {key: raterange}
			resultrating.update(case)
		else:
			raterange=[]
			case = {key: raterange}
			resultrating.update(case)
	print(resultrating)
	print(app.url_map)
	return render_template("index.html", items=items,rating=resultrating)

@app.route("/resend")
@login_required
def resend():
	send_confirmation_email(current_user.email)
	logout_user()
	flash('Confirmation email sent successfully.', 'success')
	return redirect(url_for('login'))

@app.route("/add/<id>", methods=['POST'])
def add_to_cart(id):
	if not current_user.is_authenticated:
		flash(f'You must login first!<br> <a href={url_for("login")}>Login now!</a>', 'error')
		return redirect(url_for('login'))

	item = Item.query.get(id)
	if request.method == "POST":
		quantity = request.form["quantity"]
		current_user.add_to_cart(id, quantity)
		flash(f'''{item.name} successfully added to the <a href=cart>cart</a>.<br> <a href={url_for("cart")}>view cart!</a>''','success')
		return redirect(url_for('home'))

# load and evaluate a saved model
from numpy.lib.npyio import savez_compressed
# load and evaluate a saved model
import pickle
import re
from numpy import loadtxt
from keras.models import load_model
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import os
import string
import contractions
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
lem = WordNetLemmatizer()

def strip_html(text):
    try:
        soup = BeautifulSoup(text, "html.parser")
        y = soup.get_text()
    except ValueError:
        pass
    return y

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    try:
        text = strip_html(text)
        try:
            text = remove_between_square_brackets(text)
        except ValueError:
            pass
    except ValueError:
        pass
    return text

def replace_contractions(text):
    return contractions.fix(text)
def scoreresumeall(resumetext):
	# Convert all strings to lowercase
	text = resumetext.lower()

	# Remove numbers
	text = re.sub(r'\d+', '', text)

	# Remove punctuation
	text = text.translate(str.maketrans('', '', string.punctuation))
	# Create dictionary with industrial and system engineering key terms by area
	terms = {
		'1': ['very disappointed', 'disappointed', "aren't  good", "aren't great", "isn't good", "isn't great",
			  'not too overwhelming', 'bad packaging', 'wet', 'worst product', 'wastage', 'nothing happened',
			  'stickier', 'greasy', 'aisa kyu hai ?????????', 'extremely dry', 'waste', 'nt good', 'rough', 'expensive',
			  'fail', 'worst', 'bad', 'ugly', 'baddest', 'inferior', 'poor', 'least', 'minor', 'poorest', 'secondary',
			  'unimportant', 'worst', 'incorrect', 'not right', 'atrocious', 'awful', 'cheap', 'crummy', 'dreadful',
			  'lousy', 'rough', 'sad', 'unacceptable', 'blah', 'bummer', 'diddly', 'downer', 'garbage', 'gross',
			  'imperfect', 'inferior', 'junky', 'synthetic', 'abominable', 'amiss', 'bad news', 'beastly', 'careless',
			  'cheesy', 'crappy', 'cruddy', 'defective', 'deficient', 'dissatisfactory', 'erroneous', 'fallacious',
			  'faulty', 'godawful', 'grody', 'grungy', 'icky', 'inadequate', 'incorrect', 'not good', 'off', 'raunchy',
			  'slipshod', 'stinking', 'substandard', 'the pits', 'unsatisfactory'],
		'2': ['bought', 'buy', 'care', 'contains', 'normal', 'works', 'without messing', 'improve', 'pass',
			  'average quality', 'satisfactory', 'OK', 'average', 'customer', 'right', 'true', 'acceptable', 'mediocre',
			  'moderate', 'ordinary', 'regular', 'boilerplate', 'common', 'commonplace', 'fair', 'familiar', 'garden',
			  'general', 'humdrum', 'intermediate', 'mainstream', 'medium', 'middling', 'nowhere', 'plastic',
			  'standard', 'customary', 'dime a dozen', 'everyday', 'fair to middling', 'garden-variety',
			  'middle-of-the-road', 'passable', 'run of the mill', 'so-so', 'tolerable', 'undistinguished',
			  'unexceptional', 'usual'],
		'3': ['never disappointed', 'handy', 'positive result', 'nicely', 'affordable', 'natural', 'gud', 'better',
			  'good', 'pleasing', 'sophisticated', 'positive', 'big', 'improved', 'choice', 'exceeding', 'fit',
			  'preferred', 'sharpened', 'sophisticated', 'surpassing', 'big', 'high quality', 'appropriate',
			  'desirable', 'fitting', 'select', 'suitable', 'useful', 'valuable', 'preferable', 'worthy'],
		'4': ['never disappointed', 'glad', 'worth try', 'really handy', 'happy', 'pleasing', 'blends like butter',
			  'easily', 'enriched', 'enjoyed', 'safe', 'thankyou', 'longlasting', 'quickly', 'gently', 'shiny',
			  'soothing', 'smooth', 'soft', 'works well', 'well', 'moisturizes', 'helps', 'reasonable', 'gentle',
			  'easily ', 'absorb', 'absorbs', 'blends', 'repairs', 'repair', 'save', 'saving', 'decent', 'reduce',
			  'fall stop', 'promote', 'friendly', 'manageable', 'very smooth', 'very light weighted', 'impressed',
			  'travel-friendly', 'liked', 'fresh', 'smooth silky', 'multipurpose', 'live without', 'baby',
			  'smooth nd shiny', 'care completely ', 'favourite', 'must buy', 'really helps', 'really nicely', 'pure',
			  'shine', 'progressively', 'moderately', 'easily absorbs', 'bst', 'professionally', 'smoother',
			  'greatlook', 'proper', 'long lasting', 'very good', 'finer', 'bigger', 'larger', 'higher',
			  'higher quality', 'cool', 'valuable', 'favorable', 'greater', 'satisfying', 'superb', 'wonderful''lucky',
			  'effective', 'beneficial', 'benevolent', 'honest', 'profitable', 'reputable', 'undecayed', 'upright',
			  'worthier', 'super', 'sound', 'fitter', 'more appropriate', 'more desirable', 'more fitting',
			  'more select', 'more suitable', 'more useful', 'more valuable', 'prominent', 'souped-up'],
		'5': ['never disappointed', 'awsm', 'highly recommend', 'must try', 'really amazing',
			  'very very light weighted', 'much smooth', 'really really amazing', 'super impressed', '❤', 'shimmery',
			  'loved', 'Highly recommended', 'worth buying', 'absolute favourite', 'five stars', 'smoothest', 'thumbs',
			  'fabulous', 'wow', 'champion', 'amazing', 'awesome', 'very very good', 'highest quality', 'exceptional',
			  'excellent', 'marvelous', 'incredible', 'biggest', 'love', 'lovely', 'bestest', 'beautiful', 'nicest',
			  'finest', 'first', 'first-rate', 'leading', 'outstanding', 'perfect', 'terrific', '10', 'ace', 'boss',
			  'capital', 'champion', 'culminating', 'nonpareil', 'optimum', 'premium', 'prime', 'primo', 'principal',
			  'super', 'superlative', 'tops', 'A-1', 'beyond compare', 'choicest', 'first-class', 'foremost',
			  'greatest', 'highest', 'incomparable', 'inimitable', 'matchless', 'number 1', 'out-of-sight', 'paramount',
			  'peerless', 'preeminent', 'sans pareil', 'second to none', 'supreme', 'transcendent', 'unequaled',
			  'unparalleled', 'unrivaled', 'unsurpassed']}

	# Initializie score counters for each area
	quality = 0
	operations = 0
	supplychain = 0
	project = 0
	data = 0
	healthcare = 0

	# Create an empty list where the scores will be stored
	scores = []

	# Obtain the scores for each area
	for area in terms.keys():

		if area == '1':
			for word in terms[area]:
				if word in text:
					quality += 1
			scores.append(quality)

		elif area == '2':
			for word in terms[area]:
				if word in text:
					operations += 1
			scores.append(operations)

		elif area == '3':
			for word in terms[area]:
				if word in text:
					supplychain += 1
			scores.append(supplychain)

		elif area == '4':
			for word in terms[area]:
				if word in text:
					project += 1
			scores.append(project)

		elif area == '5':
			for word in terms[area]:
				if word in text:
					data += 1
			scores.append(data)

		else:
			for word in terms[area]:
				if word in text:
					healthcare += 1
			scores.append(healthcare)

	# Create a data frame with the scores summary
	summary = pd.DataFrame(scores, index=terms.keys(), columns=['score']).sort_values(by='score', ascending=False)
	summary['rating'] = summary.index
	summary

	return summary

def predictrating(txt):
	basedir = os.path.abspath(os.path.dirname(__file__))
	with open(os.path.join(basedir, 'tokenizer.pickle'), 'rb') as handle:
		loaded_tokenizer = pickle.load(handle)
	txt = str(txt)
	seq = loaded_tokenizer.texts_to_sequences([txt])
	max_len = 50
	padded = pad_sequences(seq, maxlen=max_len)
	model = load_model(os.path.join(basedir, 'model.h5'))
	comment = denoise_text(txt)
	comment= replace_contractions(comment)
	comment = comment.replace('[^\w\s]',' ')
	comment= comment.lower()
	comment= comment.split(' ')
	seq = loaded_tokenizer.texts_to_sequences(comment)
	max_len=50
	padded = pad_sequences(seq, maxlen=max_len)

	# model.summary()
	Testresults = model.predict(padded)
	df = pd.DataFrame(Testresults, columns=['1', '2', '3', '4', '5'])
	a = df.iloc[0]
	result = a.idxmax()
	dresults= scoreresumeall(txt)
	print('LSTM star rating :', result,a)
	print('C star rating :', dresults['rating'][0],dresults)
	result=int(dresults['rating'][0])
	return result


@app.route('/addreviews/<int:id>',methods=['POST','GET'])
def itemreviews(id):
	item = Item.query.get(id)
	form = ReviewsForm()
	if not current_user.is_authenticated:
		flash(f'You must login first!<br> <a href={url_for("login")}>Login now!</a>', 'error')
		return redirect(url_for('login'))
	userid = current_user.id
	print(userid)
	user=User.query.get(userid)
	if form.validate_on_submit():
		ratings = predictrating(form.comments)
		if request.method == "POST":
			if form.validate_on_submit():
				new_comment = Review_item(username=user.name,
										  itemid=item.id,
										  comment=form.comments.data,
										  rating=ratings)
				db.session.add(new_comment)
				db.session.commit()
			flash('Thanks for your reviews!', 'success')
			return redirect(url_for('item', id=item.id))
	rate = []
	rating= [ ]
	reviews = []
	finalrating=[]
	temprate=[]
	resultrating = dict()
	ab=0
	reviewsdataframe= pd.DataFrame(columns=['rating'])
	resultrating = dict()
	if len(Review_item.query.filter_by(itemid=int(item.id)).all())>0:
		reviewslist = Review_item.query.filter_by(itemid=int(item.id)).all()
		for review in reviewslist:
			review = str(review)
			reviewlist = review.split(",")
			print(reviewlist)
			if reviewlist[-1] != None:
				reviewsdataframe.append([int(reviewlist[-1])])
				print(reviewsdataframe)
				rate.append([int(reviewlist[-1])])
				reviews.append(reviewlist[0]+' : '+reviewlist[-2])
		finalrating=int(np.mean(rate))
	elif len(Review_item.query.filter_by(itemid=int(item.id)).all()) == 0:
		finalrating = 0
	raterange = []
	print(finalrating)
	if finalrating > 0:
		for i in range(0, finalrating):
			raterange.append(i)
	else:
		raterange = []
	print(raterange)
	return render_template('itemreviews.html', item=item ,rating=raterange,reviews=reviews, form=form)

@app.route("/cart")
@login_required
def cart():
	price = 0
	price_ids = []
	items = []
	quantity = []
	for cart in current_user.cart:
		items.append(cart.item)
		quantity.append(cart.quantity)
		price_id_dict = {
			"price": cart.item.price_id,
			"quantity": cart.quantity,
			}
		price_ids.append(price_id_dict)
		price += cart.item.price*cart.quantity
	return render_template('cart.html', items=items, price=price, price_ids=price_ids, quantity=quantity)

@app.route('/orders')
@login_required
def orders():
	return render_template('orders.html', orders=current_user.orders)

@app.route("/remove/<id>/<quantity>")
@login_required
def remove(id, quantity):
	current_user.remove_from_cart(id, quantity)
	return redirect(url_for('cart'))

@app.route('/item/<int:id>')
def item(id):
	item = Item.query.get(id)
	rate = []
	rating= [ ]
	reviews = []
	finalrating=[]
	temprate=[]
	resultrating = dict()
	ab=0
	reviewsdataframe= pd.DataFrame(columns=['rating'])
	resultrating = dict()
	if len(Review_item.query.filter_by(itemid=int(item.id)).all())>0:
		reviewslist = Review_item.query.filter_by(itemid=int(item.id)).all()
		for review in reviewslist:
			review = str(review)
			reviewlist = review.split(",")
			print(reviewlist)
			if reviewlist[-1] != None:
				reviewsdataframe.append([int(reviewlist[-1])])
				print(reviewsdataframe)
				rate.append([int(reviewlist[-1])])
				reviews.append(reviewlist[0]+' : '+reviewlist[-2])
		finalrating=int(np.mean(rate))
	elif len(Review_item.query.filter_by(itemid=int(item.id)).all()) == 0:
		finalrating = 0
	raterange = []
	print(finalrating)
	if finalrating > 0:
		for i in range(0, finalrating):
			raterange.append(i)
	else:
		raterange = []
	print(raterange)
	form = ReviewsForm()
	return render_template('item.html', item=item ,rating=raterange,reviews=reviews,form=form)

@app.route('/search')
def search():
	query = request.args['query']
	search = "%{}%".format(query)
	items = Item.query.filter(Item.name.like(search)).all()
	rate = []
	rating= [ ]
	reviews = []
	finalrating=[]
	temprate=[]
	resultrating = dict()
	ab=0
	reviewsdataframe= pd.DataFrame(columns=['rating'])
	resultrating = dict()
	ab=0
	for item in items:
		print(item)
		ab=ab+1
		key= item.name
		rate=[]
		print(len(Review_item.query.filter_by(itemid=int(item.id)).all()))
		if len(Review_item.query.filter_by(itemid=int(item.id)).all())>0:
			reviewslist = Review_item.query.filter_by(itemid=int(item.id)).all()
			for review in reviewslist:
				review = str(review)
				reviewlist = review.split(",")
				print(reviewlist)
				if reviewlist[-1] != None:
					reviewsdataframe.append([int(reviewlist[-1])])
					print(reviewsdataframe)
					rate.append([int(reviewlist[-1])])
			finalrating = int(np.mean(rate))
		elif len(Review_item.query.filter_by(itemid=int(item.id)).all()) == 0:
			finalrating = 0
		raterange=[]
		print(finalrating)
		if finalrating > 0:
			for i in range(0, finalrating):
				raterange.append(i)
			case = {key: raterange}
			resultrating.update(case)
		else:
			raterange = []
			case = {key: raterange}
			resultrating.update(case)
	print(resultrating)
	return render_template('home.html', items=items, rating=resultrating,search=True, query=query)

# stripe stuffs
@app.route('/payment_success')
def payment_success():
	return render_template('success.html')

@app.route('/payment_failure')
def payment_failure():
	return render_template('failure.html')

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
	data = json.loads(request.form['price_ids'].replace("'", '"'))
	try:
		checkout_session = stripe.checkout.Session.create(
			client_reference_id=current_user.id,
			line_items=data,
			payment_method_types=[
			  'card',
			],
			mode='payment',
			success_url=url_for('payment_success', _external=True),
			cancel_url=url_for('payment_failure', _external=True),
		)
		return redirect(checkout_session.url, code=303)
	except Exception as e:
		return str(e)
		return redirect(checkout_session.url, code=303)

@app.route('/stripe-webhook', methods=['POST'])
def stripe_webhook():

	if request.content_length > 1024*1024:
		print("Request too big!")
		abort(400)

	payload = request.get_data()
	sig_header = request.environ.get('HTTP_STRIPE_SIGNATURE')
	ENDPOINT_SECRET = os.environ.get('ENDPOINT_SECRET')
	event = None

	try:
		event = stripe.Webhook.construct_event(
		payload, sig_header, ENDPOINT_SECRET
		)
	except ValueError as e:
		# Invalid payload
		return {}, 400
	except stripe.error.SignatureVerificationError as e:
		# Invalid signature
		return {}, 400

	if event['type'] == 'checkout.session.completed':
		session = event['data']['object']

		# Fulfill the purchase...
		fulfill_order(session)

	# Passed signature verification
	return {}, 200


