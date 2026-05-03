// Package anthropic adapts Anthropic's Messages API as a fugue.Agent.
package anthropic

import (
	"context"
	"iter"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/inancsege/fugue"
)

// New returns a fugue.Agent that calls Anthropic's Messages API.
//
// model is the Anthropic model identifier, e.g. "claude-sonnet-4-6".
// At minimum, one of WithAPIKey or the ANTHROPIC_API_KEY environment
// variable must provide credentials.
//
// New("") panics — an empty model is a programming bug.
func New(model string, opts ...Option) fugue.Agent {
	if model == "" {
		panic("anthropic: New() requires a non-empty model")
	}
	cfg := config{
		maxTokens: 1024,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.client == nil {
		var clientOpts []option.RequestOption
		if cfg.apiKey != "" {
			clientOpts = append(clientOpts, option.WithAPIKey(cfg.apiKey))
		}
		c := sdk.NewClient(clientOpts...)
		cfg.client = &c
	}
	return &agent{model: model, cfg: cfg}
}

// Option configures an agent at construction.
type Option func(*config)

type config struct {
	apiKey       string
	systemPrompt string
	maxTokens    int
	temperature  *float64
	client       *sdk.Client
}

// WithAPIKey sets the API key. Overrides ANTHROPIC_API_KEY from the environment.
func WithAPIKey(key string) Option { return func(c *config) { c.apiKey = key } }

// WithSystemPrompt sets the system prompt. Overrides any leading RoleSystem
// messages in the input (which would otherwise be concatenated into the system
// parameter).
func WithSystemPrompt(prompt string) Option {
	return func(c *config) { c.systemPrompt = prompt }
}

// WithMaxTokens sets the maximum tokens to generate. Defaults to 1024.
func WithMaxTokens(n int) Option { return func(c *config) { c.maxTokens = n } }

// WithTemperature sets the sampling temperature. Unset uses the SDK default.
func WithTemperature(t float64) Option { return func(c *config) { c.temperature = &t } }

// WithClient injects a custom *anthropic.Client — for tests or custom transports.
func WithClient(client *sdk.Client) Option { return func(c *config) { c.client = client } }

type agent struct {
	model string
	cfg   config
}

func (a *agent) Invoke(ctx context.Context, in []fugue.Message) ([]fugue.Message, error) {
	params, err := a.buildParams(in)
	if err != nil {
		return nil, err
	}
	resp, err := a.cfg.client.Messages.New(ctx, params)
	if err != nil {
		return nil, err
	}
	out, err := fromAPIResponse(resp)
	if err != nil {
		return nil, err
	}
	return []fugue.Message{out}, nil
}

func (a *agent) Stream(ctx context.Context, in []fugue.Message) iter.Seq2[fugue.Event[[]fugue.Message], error] {
	return func(yield func(fugue.Event[[]fugue.Message], error) bool) {
		params, err := a.buildParams(in)
		if err != nil {
			yield(fugue.Event[[]fugue.Message]{}, err)
			return
		}
		stream := a.cfg.client.Messages.NewStreaming(ctx, params)
		defer stream.Close()

		var acc sdk.Message
		for stream.Next() {
			event := stream.Current()
			if err := acc.Accumulate(event); err != nil {
				yield(fugue.Event[[]fugue.Message]{}, err)
				return
			}
			cumulative, err := fromAPIResponse(&acc)
			if err != nil {
				yield(fugue.Event[[]fugue.Message]{}, err)
				return
			}
			done := event.Type == "message_stop"
			if !yield(fugue.Event[[]fugue.Message]{
				Delta: []fugue.Message{cumulative},
				Done:  done,
			}, nil) {
				return
			}
		}
		if err := stream.Err(); err != nil {
			yield(fugue.Event[[]fugue.Message]{}, err)
		}
	}
}

// buildParams assembles MessageNewParams from the input. Shared between
// Invoke and Stream.
func (a *agent) buildParams(in []fugue.Message) (sdk.MessageNewParams, error) {
	system, body, err := splitSystem(in, a.cfg.systemPrompt)
	if err != nil {
		return sdk.MessageNewParams{}, err
	}
	apiMessages, err := toAPIMessages(body)
	if err != nil {
		return sdk.MessageNewParams{}, err
	}
	params := sdk.MessageNewParams{
		Model:     sdk.Model(a.model),
		MaxTokens: int64(a.cfg.maxTokens),
		Messages:  apiMessages,
	}
	if system != "" {
		params.System = []sdk.TextBlockParam{{Text: system}}
	}
	if a.cfg.temperature != nil {
		params.Temperature = sdk.Float(*a.cfg.temperature)
	}
	return params, nil
}
